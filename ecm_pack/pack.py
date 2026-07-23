# pack.py
# Pack 耦合求解器：把 ECM 电芯 + 电路(MNA) + 热网络 用「双步循环」耦合起来。
#
# 每一时间步执行：
#   1) 由当前电芯状态算出 E_k（R0 之后等效电动势）与 R0_k，写入网表；
#   2) 解 MNA 电路，得到每支路电流 I_k（= 各电芯电流）；
#   3) 用 I_k 推进每个电芯的电学状态(SoC/RC/扩散)，并算产热；
#   4) 用产热 + 自定义电芯间导热 + 对流，推进热网络，更新每个电芯温度；
#   5) 记录端口与每芯输出。
#
# 这正是 liionpack「更新网表 → 解电路 → 步进电化学模型 → 更新热」的范式，
# 只不过把黑盒 PyBaMM 模型替换成了我们可定制的 ECM 电芯。
import numpy as np

from .circuit import solve_circuit


class Pack:
    def __init__(
        self,
        cells,
        netlist,
        thermal=None,
        v_cut_lower=2.0,
        v_cut_upper=4.4,
        cell_current_sign=-1.0,
        active=None,
    ):
        if len(cells) < 1:
            raise ValueError("至少需要一个电芯")
        self.cells = list(cells)
        self.netlist = netlist
        self.thermal = thermal
        self.N_total = len(cells)          # 全部电芯数（状态始终保留）
        self.N = self.N_total
        self.v_cut_lower = v_cut_lower
        self.v_cut_upper = v_cut_upper
        self.cell_current_sign = cell_current_sign
        # active：当前接入网表的电芯在 self.cells 中的下标，顺序与网表 V 元素一致
        # 拓扑切换时只改 active 与 netlist，电芯自身状态(SoC/RC/T)完全保留
        if active is None:
            active = list(range(self.N_total))
        self.active = list(active)
        self._set_netlist_maps(netlist)
        if len(self.v_rows) != len(self.active):
            raise ValueError(
                f"网表 V 元素数({len(self.v_rows)}) 与激活电芯数({len(self.active)}) 不一致"
            )
        # 初始温度同步到电芯
        if thermal is not None:
            for k, c in enumerate(self.cells):
                c.T = thermal.T[k]

    def _set_netlist_maps(self, netlist):
        """从网表提取 V / R0 元素行号（顺序即接入电芯顺序）。"""
        df = netlist.df
        self.netlist = netlist
        self.v_rows = np.where(df["desc"].str[0] == "V")[0]
        self.ri_rows = np.where(df["desc"].str[:2] == "R0")[0]

    def set_topology(self, netlist, active):
        """
        随时/定时切换整包拓扑（可重构电池包核心能力）。

        仅替换网表与「接入电芯映射」，不触碰任何电芯的 SoC / RC / 温度状态，
        因此：
          - 被断开的电芯保持原状态「待机」；
          - 新并入的电芯按自身状态立即参与电路（并联瞬间由 MNA 自然产生环流/浪涌）。
        接入顺序 active[k] 必须与网表第 k 个 V 元素对应的电芯一致。

        参数
        ----
        netlist : Netlist   新拓扑网表
        active  : list[int] 新拓扑中接入的电芯在 self.cells 中的下标
        """
        active = list(active)
        n_v = int((netlist.df["desc"].str[0] == "V").sum())
        if n_v != len(active):
            raise ValueError(
                f"新网表 V 元素数({n_v}) 与 active 数({len(active)}) 不一致"
            )
        self.active = active
        self._set_netlist_maps(netlist)
        # 立即把当前电芯状态写入新网表，保证切换后首步 MNA 解正确
        for k, idx in enumerate(self.active):
            cell = self.cells[idx]
            self.netlist.df.at[self.v_rows[k], "value"] = cell.voltage_behind_R0()
            Td = cell.T - 273.15
            R0 = cell.spec.R0(Td, 0.0, cell.soc)
            self.netlist.df.at[self.ri_rows[k], "value"] = R0
        # 温度同步（新接入电芯的温度对齐到电芯对象）
        if self.thermal is not None:
            for k, idx in enumerate(self.active):
                self.thermal.T[idx] = self.cells[idx].T

    def _refresh_R0(self, I_prev):
        for k, idx in enumerate(self.active):
            cell = self.cells[idx]
            Td = cell.T - 273.15
            R0 = cell.spec.R0(Td, I_prev[k], cell.soc)
            self.netlist.df.at[self.ri_rows[k], "value"] = R0

    def solve(
        self,
        dt,
        control,
        control_type="current",
        n_steps=None,
        record_every=1,
        topology_events=None,
        switch_callback=None,
    ):
        """
        求解整包（支持运行中随时/定时切换拓扑）。

        参数
        ----
        dt, control, control_type, n_steps, record_every : 同前。
        topology_events : list, 可选
            定时切换列表，元素为 (t_switch, netlist, active) 或
            (t_switch, factory) 其中 factory()->(netlist, active)。
            到达 t_switch 时自动调用 set_topology 切换（电芯状态保留）。
        switch_callback : callable, 可选
            事件驱动切换：每个时间步调用 callback(t, pack)，若返回
            (netlist, active) 则立即切换。用于「随时/按工况」切换。

        返回
        ----
        output : dict（每芯数组按 self.cells 的全长 N_total 对齐，
                未接入的电芯电流记 0、状态保持冻结）。
        """
        control = np.atleast_1d(np.asarray(control, dtype=float))
        if n_steps is None:
            if control.size == 1:
                raise ValueError("标量 control 需要提供 n_steps")
            n_steps = control.size
        else:
            if control.size == 1:
                control = np.full(n_steps, control[0])
            elif control.size != n_steps:
                control = np.resize(control, n_steps)

        N = self.N_total
        active_set = set(self.active)

        # 初始化：把当前激活电芯的 E_k / R0_k 写入网表
        self._refresh_R0(np.zeros(len(self.active)))
        for k, idx in enumerate(self.active):
            self.netlist.df.at[self.v_rows[k], "value"] = self.cells[idx].voltage_behind_R0()

        # 初始电路解，得到 t=0 端口量与激活电芯电流
        try:
            if control_type == "current":
                Vn0, Ib0, It0, Vt0, Pt0 = solve_circuit(self.netlist, current=control[0])
            else:
                Vn0, Ib0, It0, Vt0, Pt0 = solve_circuit(self.netlist, power=control[0])
            Ic0 = self.cell_current_sign * Ib0
        except Exception:
            Vn0, It0, Vt0, Pt0, Ic0 = (np.zeros(N + 1), 0.0, 0.0, 0.0, np.zeros(N))
        Ic0_full = np.zeros(N)
        for k, idx in enumerate(self.active):
            Ic0_full[idx] = Ic0[k]

        n_rec = (n_steps // record_every) + 1
        out = {
            "Time [s]": np.zeros(n_rec),
            "Pack current [A]": np.zeros(n_rec),
            "Pack terminal voltage [V]": np.zeros(n_rec),
            "Pack power [W]": np.zeros(n_rec),
            "Cell current [A]": np.zeros((n_rec, N)),
            "Cell terminal voltage [V]": np.zeros((n_rec, N)),
            "Cell SoC": np.zeros((n_rec, N)),
            "Cell current abs [A]": np.zeros((n_rec, N)),
            "Cell temperature [K]": np.zeros((n_rec, N)),
            "Cell internal resistance [Ohm]": np.zeros((n_rec, N)),
            "Topology changes [s]": [],
        }

        rec_i = 0

        def record(t, V_node, I_term, V_term, P_term, I_cell, V_cell, soc, T, Rint, topo=False):
            nonlocal rec_i
            out["Time [s]"][rec_i] = t
            out["Pack current [A]"][rec_i] = I_term
            out["Pack terminal voltage [V]"][rec_i] = V_term
            out["Pack power [W]"][rec_i] = P_term
            out["Cell current [A]"][rec_i] = I_cell
            out["Cell terminal voltage [V]"][rec_i] = V_cell
            out["Cell SoC"][rec_i] = soc
            out["Cell current abs [A]"][rec_i] = np.abs(I_cell)
            out["Cell temperature [K]"][rec_i] = T
            out["Cell internal resistance [Ohm]"][rec_i] = Rint
            if topo:
                out["Topology changes [s]"].append(t)
            rec_i += 1

        # 初始快照 (t=0)
        self._snapshot(0.0, record, Vn0, It0, Vt0, Pt0, Ic0_full)

        # 拓扑事件簿：记录已应用的定时事件
        applied = [False] * (len(topology_events) if topology_events else 0)

        def maybe_switch(t):
            switched = False
            # 定时事件
            if topology_events:
                for i, ev in enumerate(topology_events):
                    if applied[i]:
                        continue
                    t_switch = ev[0]
                    if t >= t_switch:
                        nl, act = self._resolve_event(ev)
                        self.set_topology(nl, act)
                        applied[i] = True
                        switched = True
            # 回调事件（随时/按工况）
            if switch_callback is not None:
                res = switch_callback(t, self)
                if res is not None:
                    nl, act = res if isinstance(res, tuple) else (res, list(range(len(self.cells))))
                    self.set_topology(nl, act)
                    switched = True
            return switched

        terminated = False
        term_step = n_steps
        for s in range(n_steps):
            t = (s + 1) * dt
            topo = maybe_switch((s + 1) * dt)  # 以步末时刻判断是否切换
            if topo:
                # 切换后用新网表重解一次初始端口量（保持记录自洽）
                try:
                    if control_type == "current":
                        Vn0, Ib0, It0, Vt0, Pt0 = solve_circuit(self.netlist, current=control[s])
                    else:
                        Vn0, Ib0, It0, Vt0, Pt0 = solve_circuit(self.netlist, power=control[s])
                    Ic0 = self.cell_current_sign * Ib0
                except Exception:
                    Vn0, Ib0, It0, Vt0, Ic0 = (np.zeros(N + 1), np.zeros(1), 0.0, 0.0, np.zeros(len(self.active)))
                Ic0_full = np.zeros(N)
                for k, idx in enumerate(self.active):
                    Ic0_full[idx] = Ic0[k]
                active_set = set(self.active)

            ctrl = control[s]
            if control_type == "current":
                V_node, I_batt, I_term, V_term, P_term = solve_circuit(self.netlist, current=ctrl)
            else:
                V_node, I_batt, I_term, V_term, P_term = solve_circuit(self.netlist, power=ctrl)
            I_cell_active = self.cell_current_sign * I_batt  # 仅激活电芯

            # 全量数组（未接入电芯：电流 0、状态冻结）
            I_cell = np.zeros(N)
            V_cell = np.zeros(N)
            soc = np.zeros(N)
            T = np.zeros(N)
            Rint = np.zeros(N)
            Q = np.zeros(N)
            for k, idx in enumerate(self.active):
                cell = self.cells[idx]
                I = I_cell_active[k]
                R0 = cell.step_electrical(I, dt)
                Vt = cell.terminal_voltage(R0, I)
                I_cell[idx] = I
                V_cell[idx] = Vt
                soc[idx] = cell.soc
                T[idx] = cell.T
                Rint[idx] = abs((cell.voltage_behind_R0() + I * R0 - Vt) / (I + 1e-12))
                Q[idx] = cell.heat(I, R0)
            # 未接入电芯：保持冻结状态
            for idx in range(N):
                if idx not in active_set:
                    I_cell[idx] = 0.0
                    V_cell[idx] = self.cells[idx].voltage_behind_R0()
                    soc[idx] = self.cells[idx].soc
                    T[idx] = self.cells[idx].T
                    Rint[idx] = self.cells[idx].spec.R0(self.cells[idx].T - 273.15, 0.0, self.cells[idx].soc)

            # 热网络推进（含自定义电芯间导热；未接入电芯产热为 0）
            if self.thermal is not None:
                self.thermal.step(Q, dt, t=t)
                for idx in range(N):
                    self.cells[idx].T = self.thermal.T[idx]
                T = self.thermal.T.copy()

            # 写回网表供下一步求解
            for k, idx in enumerate(self.active):
                self.netlist.df.at[self.v_rows[k], "value"] = self.cells[idx].voltage_behind_R0()
            self._refresh_R0(I_cell_active)

            if (s + 1) % record_every == 0 or s == n_steps - 1:
                self._snapshot(t, record, V_node, I_term, V_term, P_term, I_cell, V_cell, soc, T, Rint, topo)

            if np.any(V_cell[self.active] < self.v_cut_lower) or np.any(V_cell[self.active] > self.v_cut_upper):
                terminated = True
                term_step = s + 1
                break

        out["Terminated"] = terminated
        out["Term step"] = term_step
        out["_terminated"] = terminated
        out["_term_step"] = term_step
        used = rec_i
        for key in [
            "Time [s]", "Pack current [A]", "Pack terminal voltage [V]", "Pack power [W]",
            "Cell current [A]", "Cell terminal voltage [V]", "Cell SoC",
            "Cell current abs [A]", "Cell temperature [K]", "Cell internal resistance [Ohm]",
        ]:
            out[key] = out[key][:used]
        return out

    @staticmethod
    def _resolve_event(ev):
        if len(ev) == 3:
            return ev[1], ev[2]
        elif len(ev) == 2:
            return ev[1]()
        else:
            raise ValueError("topology_events 元素格式应为 (t, netlist, active) 或 (t, factory)")

    def _snapshot(self, t, record, V_node, I_term, V_term, P_term, I_cell, V_cell=None, soc=None, T=None, Rint=None, topo=False):
        if V_cell is None:
            V_cell = np.array([c.voltage_behind_R0() for c in self.cells])
            soc = np.array([c.soc for c in self.cells])
            T = np.array([c.T for c in self.cells])
            Rint = np.array([c.spec.R0(c.T - 273.15, 0.0, c.soc) for c in self.cells])
        record(t, V_node, I_term, V_term, P_term, I_cell, V_cell, soc, T, Rint, topo)

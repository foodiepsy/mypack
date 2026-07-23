# circuit.py
# 电路层：网表(Netlist) + 修正节点法(MNA)求解器 + 串并联拓扑生成器
#
# 耦合思路（与 liionpack 同源）：
#   每个电芯在网表中表示为「电压源 V_k（值=E_k，R0 之后的等效电动势）
#   + 串联电阻 Ri_k（欧姆内阻 R0）」。整包用 MNA 求解出每个支路的电流 I_k，
#   再把这个电流回灌给对应的 ECM 电芯做一步电学/热学推进。
import numpy as np
import pandas as pd


class Netlist:
    """电路网表。元素字典列表：{'desc','node1','node2','value'}。
    desc 首字母：'V'=电压源(电芯), 'R'=电阻, 'I'=电流源(整包负载)。"""

    def __init__(self, elements):
        self.df = pd.DataFrame(elements)
        if not {"desc", "node1", "node2", "value"}.issubset(self.df.columns):
            raise ValueError("元素需含 desc, node1, node2, value")

    @classmethod
    def from_dict(cls, desc, node1, node2, value):
        return cls(
            [
                {"desc": d, "node1": n1, "node2": n2, "value": v}
                for d, n1, n2, v in zip(desc, node1, node2, value)
            ]
        )

    def __len__(self):
        return len(self.df)

    @property
    def n_cells(self):
        return int((self.df["desc"].str[0] == "V").sum())


def solve_circuit(netlist, current=None, power=None):
    """
    求解修正节点法(MNA)线性方程 A·x = z。
    返回 (V_node, I_batt, terminal_current, terminal_voltage, terminal_power)。

    V_node   : 各节点电压（含地节点 0）
    I_batt   : 每个电压源支路电流（与网表中 V 元素顺序一致），即各电芯电流
    terminal_*: 整包端口的电流/电压/功率
    """
    df = netlist.df
    desc = df["desc"].values.astype(str)
    d0 = np.array([d[0] for d in desc])
    I_map = d0 == "I"
    R_map = d0 == "R"
    V_map = d0 == "V"
    node1 = df["node1"].values.astype(int)
    node2 = df["node2"].values.astype(int)
    value = df["value"].values.astype(float)

    # 输入校验：所有电阻必须为正，避免 1/value 除零把 inf/nan 注入 G 矩阵
    if np.any(value[R_map] <= 0):
        bad_rows = np.where(R_map)[0][value[R_map] <= 0]
        raise ValueError(f"电阻必须为正(R>0)，以下网表行非法: {bad_rows.tolist()}")

    # 节点编号：0 为地；内部矩阵索引 0..n-1（地 = -1）
    # 注意：节点号不连续（含电芯私有节点），n 必须取最大节点号
    n = int(max(node1.max(), node2.max()))  # 最大节点号
    m = int(V_map.sum())  # 电压源数
    G = np.zeros((n, n))
    B = np.zeros((n, m))
    D = np.zeros((m, m))
    i = np.zeros((n, 1))
    e = np.zeros((m, 1))

    n1 = node1 - 1
    n2 = node2 - 1

    # 电阻 -> G 矩阵
    for idx in np.where(R_map)[0]:
        a, b, g = n1[idx], n2[idx], 1.0 / value[idx]
        if a >= 0:
            G[a, a] += g
        if b >= 0:
            G[b, b] += g
        if a >= 0 and b >= 0:
            G[a, b] -= g
            G[b, a] -= g

    # 电压源 -> B / e
    vs_idx = np.where(V_map)[0]
    for cnt, idx in enumerate(vs_idx):
        a, b = n1[idx], n2[idx]
        if a >= 0:
            B[a, cnt] += 1.0
        if b >= 0:
            B[b, cnt] -= 1.0
        e[cnt, 0] = value[idx]

    A = np.block([[G, B], [B.T, D]])

    # 控制方式
    control_args = sum(arg is not None for arg in (current, power))
    if control_args == 2:
        raise ValueError("current 与 power 只能指定一个")

    Terminal_Node = np.array(df[I_map].node1, dtype=int)

    if control_args == 0:
        current = value[I_map]
    if current is not None:
        cur = np.atleast_1d(np.asarray(current, dtype=float))
        if len(cur) == 1 and I_map.sum() > 1:
            cur = np.full(I_map.sum(), cur[0])
        # 注入到电流源支路的端节点
        ni = n1[I_map]
        nj = n2[I_map]
        for k in range(len(cur)):
            if ni[k] >= 0:
                i[ni[k], 0] -= cur[k]
            if nj[k] >= 0:
                i[nj[k], 0] += cur[k]
        z = np.vstack([i, e])
        X = np.linalg.solve(A, z).flatten()
        I_batt = X[n:]
        V_node = np.zeros(n + 1)
        V_node[1:] = X[:n]
        terminal_current = cur
    else:  # power 控制：线性电路端口为 Thevenin 仿射关系 V(I)=V_oc-R_eq·I，
        # 故 P=V·I=V_oc·I-R_eq·I² 有**闭式精确解**（无需脆弱迭代）。
        m_src = int(I_map.sum())
        ni = n1[I_map]
        nj = n2[I_map]

        def _solve_with_current(cur):
            i = np.zeros((n, 1))
            for k in range(len(cur)):
                if ni[k] >= 0:
                    i[ni[k], 0] -= cur[k]
                if nj[k] >= 0:
                    i[nj[k], 0] += cur[k]
            z = np.vstack([i, e])
            X = np.linalg.solve(A, z).flatten()
            Vn = np.zeros(n + 1)
            Vn[1:] = X[:n]
            return X, Vn

        # V_oc：全部负载电流=0；R_eq：逐源单位电流探针（对角 Thevenin 等效）
        _, Vn_oc = _solve_with_current(np.zeros(m_src))
        V_oc = Vn_oc[Terminal_Node]
        R_eq = np.zeros(m_src)
        for k in range(m_src):
            probe = np.zeros(m_src)
            probe[k] = 1.0
            _, Vn_p = _solve_with_current(probe)
            R_eq[k] = V_oc[k] - Vn_p[Terminal_Node[k]]

        power_arr = np.atleast_1d(np.asarray(power, dtype=float))
        if power_arr.size == 1 and m_src > 1:
            power_arr = np.full(m_src, power_arr[0])
        cur = np.zeros(m_src)
        for k in range(m_src):
            P = power_arr[k] if k < power_arr.size else power_arr[-1]
            if R_eq[k] <= 0:
                raise ValueError(
                    f"功率控制：第{k}个负载支路等效内阻非正 (R_eq={R_eq[k]:.4g})，无法求解"
                )
            Pmax = V_oc[k] ** 2 / (4.0 * R_eq[k])
            if P > Pmax + 1e-6:
                raise ValueError(
                    f"功率控制：需求 {P:.4g}W 超出第{k}个负载最大可输出功率 "
                    f"Pmax={Pmax:.4g}W（不可行）"
                )
            disc = max(V_oc[k] ** 2 - 4.0 * R_eq[k] * P, 0.0)
            # 取低电流(高电压)的物理根；另一根是高电流低电压的病态分支
            cur[k] = (V_oc[k] - np.sqrt(disc)) / (2.0 * R_eq[k])
        # 用精确线性解得到自洽的 V_node / I_batt
        X, V_node = _solve_with_current(cur)
        I_batt = X[n:]
        terminal_current = cur

    terminal_voltage = V_node[Terminal_Node]
    terminal_power = terminal_voltage * terminal_current
    return V_node, I_batt, terminal_current, terminal_voltage, terminal_power


def setup_circuit(n_series, n_parallel, Rbus=0.0):
    """
    自动生成 n_series 串、n_parallel 并 的网表（标准网格拓扑）。
    节点布局：负极(整包负端)=地 node 0；每串级 s 占节点 s(负)->s+1(正)，
    s 级的正端即 s+1 级的负端（串联）。正极节点 = nS。
    每级内 nP 个电芯并联在节点 s 与 s+1 之间，每芯 = 电压源 V_k + 串联 R0_k。
    整包电流源 I 接在正极(nS)与地(0)之间（负载跨接整包两端）。

    返回 (netlist, v_rows, ri_rows)：v_rows/ri_rows 给出每个电芯对应的
    网表 V / R0 元素行号（顺序与 cells 列表一一对应）。
    """
    elements = []
    nS, nP = int(n_series), int(n_parallel)
    pos_node = nS  # 正极节点
    cell_priv_base = nS + 1  # 电芯私有节点起点
    cell_idx = 0
    for s in range(nS):
        a_s = s          # 本串级负端
        b_s = s + 1      # 本串级正端
        left = a_s
        if Rbus > 0:
            bus_node = cell_priv_base + s * nP + nP
            elements.append({"desc": "Rb", "node1": a_s, "node2": bus_node, "value": Rbus})
            left = bus_node
        for k in range(nP):
            priv = cell_priv_base + s * nP + k
            # 电压源 E_k：正极接电芯正端 b_s（MNA 约定 V_node1 - V_node2 = E）
            elements.append({"desc": f"V{cell_idx}", "node1": b_s, "node2": priv, "value": 0.0})
            # 串联欧姆内阻 R0_k：连接私有节点 priv 与电芯负端 a_s
            elements.append({"desc": f"R0{cell_idx}", "node1": priv, "node2": a_s, "value": 1e-3})
            cell_idx += 1
    # 整包电流源（负载）：正极 -> 地
    elements.append({"desc": "I", "node1": pos_node, "node2": 0, "value": 0.0})

    netlist = Netlist(elements)
    v_rows = np.where(netlist.df["desc"].str[0] == "V")[0]
    ri_rows = np.where(netlist.df["desc"].str[:2] == "R0")[0]
    return netlist, v_rows, ri_rows


def setup_two_group(n_series, Rbus=0.0):
    """
    构造「单组 nS」与「两组 nS 并联（nS2P）」两套网表，用于可重构电池包
    的拓扑热切换演示（如：一组 8S 工作 10min，另一组 8S 随后并入并联）。

    电芯编号约定：group A = 0..nS-1（先工作），group B = nS..2nS-1（随后并入）。
    两套网表都使用 **相同节点布局**（负极 node 0、正极 node nS），因此两组在
    电气上天然「并联跨接整包两端」，切换时 MNA 自动产生组间环流/浪涌。

    返回 (netlist_solo, active_solo, netlist_par, active_par)：
      netlist_solo : 仅 group A 的 nS 串联网表
      active_solo  : [0..nS-1]
      netlist_par  : group A 与 group B 各 nS 串联、两组并联的网表
      active_par   : [0..2nS-1]（V 元素顺序即电芯自然编号顺序）
    """
    nS = int(n_series)
    pos_node = nS
    # 私有节点紧贴串级节点 0..nS 之后、**连续编号**，避免 MNA 出现孤立节点(奇异矩阵)。
    # 单组：私有节点 nS+1 .. 2nS；双组：私有节点 nS+1 .. 3nS（A、B 交错，无空洞）。

    # ---- 单组：cells 0..nS-1 串联 ----
    el_solo = []
    for s in range(nS):
        priv = nS + 1 + s
        el_solo.append({"desc": f"V{s}", "node1": s + 1, "node2": priv, "value": 0.0})
        el_solo.append({"desc": f"R0{s}", "node1": priv, "node2": s, "value": 1e-3})
    el_solo.append({"desc": "I", "node1": pos_node, "node2": 0, "value": 0.0})
    nl_solo = Netlist(el_solo)
    active_solo = list(range(nS))

    # ---- 双组并联：group A(0..nS-1) 与 group B(nS..2nS-1) 各自 nS 串联，两组并联 ----
    el_par = []
    for s in range(nS):
        # group A 第 s 级：私有节点 nS+1+2s
        privA = nS + 1 + 2 * s
        el_par.append({"desc": f"V{s}", "node1": s + 1, "node2": privA, "value": 0.0})
        el_par.append({"desc": f"R0{s}", "node1": privA, "node2": s, "value": 1e-3})
        # group B 第 s 级（cell 编号 nS+s）：私有节点 nS+2+2s
        cellB = nS + s
        privB = nS + 2 + 2 * s
        el_par.append({"desc": f"V{cellB}", "node1": s + 1, "node2": privB, "value": 0.0})
        el_par.append({"desc": f"R0{cellB}", "node1": privB, "node2": s, "value": 1e-3})
    el_par.append({"desc": "I", "node1": pos_node, "node2": 0, "value": 0.0})
    nl_par = Netlist(el_par)
    # active 必须与网表 V 元素顺序一致：每级先 group A(s) 后 group B(nS+s)
    active_par = []
    for s in range(nS):
        active_par.append(s)          # V_s 对应 cell s（group A 第 s 级）
        active_par.append(nS + s)     # V_{nS+s} 对应 cell nS+s（group B 第 s 级）

    return nl_solo, active_solo, nl_par, active_par


def setup_series_bypass(n_total, bypass_idx=None):
    """
    构造 n_total 个电芯的**串联**网表；若给定 bypass_idx，则把该电芯旁路
    （移除其 V+R0，并用近零电阻短接其两端 node k 与 node k+1）。

    用途：故障容错重构——某芯快到截止电压时，BMS 自动旁路它，整包以
    (n-1) 串继续工作，而不是整体停机。

    节点保持原始串级编号 0..n（被旁路电芯的位置用短接电阻填补），因此
    其余电芯节点不变、私有节点紧贴其后连续编号，MNA 不会出现孤立节点。

    返回 (netlist, active)：active 顺序与网表 V 元素顺序一致（= 未被旁路的
    电芯下标列表）。
    """
    n = int(n_total)
    remaining = [i for i in range(n) if i != bypass_idx] if bypass_idx is not None \
        else list(range(n))
    pos_node = n  # 负载仍跨接 node n 与地 node 0
    priv_base = n + 1  # 私有节点紧贴串级节点之后，连续
    els = []
    for cnt, cell_idx in enumerate(remaining):
        a = cell_idx            # 该电芯串级负端
        b = cell_idx + 1        # 该电芯串级正端（使用原始编号，旁路后其余电芯节点不变）
        priv = priv_base + cnt
        els.append({"desc": f"V{cell_idx}", "node1": b, "node2": priv, "value": 0.0})
        els.append({"desc": f"R0{cell_idx}", "node1": priv, "node2": a, "value": 1e-3})
    if bypass_idx is not None:
        # 短接被旁路电芯两端（其原本占据 node k .. k+1）
        els.append({"desc": "Rbp", "node1": bypass_idx, "node2": bypass_idx + 1, "value": 1e-4})
    els.append({"desc": "I", "node1": pos_node, "node2": 0, "value": 0.0})
    return Netlist(els), remaining

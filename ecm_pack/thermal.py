# thermal.py
# 热网络层：每电芯集总热容 + 对环境对流 + 自定义电芯间导热

# 温度演化（每电芯）：
# C_th_i · dT_i/dt = Q_i                      (电芯自身产热)
# + Σ_j G_ij (T_j - T_i)  (电芯间导热，用户自定义)
# - h_i   (T_i - T_amb)   (对环境对流换热)

# 这正是 PyBaMM ECM 双集总(电芯+夹具)思路的推广：这里把"夹具/邻居"
# 直接建模为任意拓扑的导热网络，满足「自定义电池间相互导热」需求。
import numpy as np
class ThermalNetwork:
    """
    参数
    ----
    n_cells : int
    C_th : float 或 array_like [J/K]
        每电芯热容。
    h : float 或 array_like [W/K], 可选
        每电芯到环境的对流换热系数。默认 0（绝热）。
    T_amb : float 或 callable(t)->K, 可选
        环境温度，可随时间变化（如整车热管理曲线）。
    conduction : None / list[(i,j,G_ij)] / (n,n) ndarray, 可选
        电芯间导热。None=无相互导热；
        list 形式给出每条导热带 (i,j,导热系数)；
        ndarray 形式直接给出完整的对称导热矩阵 G。
    interface_resistance : list of (i, j, R_th, A_contact), 可选
        层间热接触电阻。每个元素为 (电芯i, 电芯j, R_th[K·m²/W], 接触面积A[m²])。
        自动换算为等效导热 G = A/R_th，叠加到 conduction 中。
        典型值：导热硅胶垫 R_th≈0.55 K·cm²/W，气隙 R_th≈20100 K·cm²/W。
    """
    def __init__(self, n_cells, C_th, h=None, T_amb=298.15, conduction=None,
                 interface_resistance=None, T_init=298.15):
        self.n = int(n_cells)
        self.C_th = np.atleast_1d(np.asarray(C_th, dtype=float))
        if self.C_th.size == 1:
            self.C_th = np.full(self.n, self.C_th.item())
        self.h = np.atleast_1d(np.asarray(h if h is not None else 0.0, dtype=float))
        if self.h.size == 1:
            self.h = np.full(self.n, self.h.item())
        self.T_amb = T_amb
        self.T = np.full(self.n, float(T_init))
        self._t = 0.0  # 内部时间累加器，供 callable(T(t)) 型环境温度使用

        # 将层间热接触电阻换算为等效导热，与 conduction 合并
        if interface_resistance is not None:
            if conduction is None:
                conduction = []
            elif isinstance(conduction, np.ndarray):
                raise ValueError(
                    "interface_resistance 与 ndarray 形式的 conduction 不能同时使用"
                )
            else:
                conduction = list(conduction)
            for entry in interface_resistance:
                i, j, R_th = entry[0], entry[1], float(entry[2])
                A = float(entry[3]) if len(entry) >= 4 else None
                if A is None or A <= 0:
                    raise ValueError(
                        f"interface_resistance 条目 ({i},{j},R_th) 需提供有效接触面积 A"
                    )
                G = A / R_th if R_th > 0 else 1e9
                conduction.append((i, j, G))
        self._build_conduction(conduction)

    def _build_conduction(self, conduction):
        n = self.n
        K = np.zeros((n, n))
        if conduction is None:
            pass
        elif isinstance(conduction, np.ndarray):
            if conduction.shape != (n, n):
                raise ValueError("导热矩阵形状须为 (n_cells, n_cells)")
            K = conduction.astype(float).copy()
        else:
            for i, j, g in conduction:
                K[i, j] += g
                K[j, i] += g  # 对称
        # 对角线 = 该行非对角元素之和的负值，使得
        #   Σ_j K_ij T_j = Σ_j g_ij (T_j - T_i)
        # 即正确的「导热进入 i = 邻居导热差之和」
        row_sum = K.sum(axis=1)
        np.fill_diagonal(K, -row_sum)
        self.K = K

    def ambient(self, t=0.0):
        return self.T_amb(t) if callable(self.T_amb) else float(self.T_amb)

    def step(self, Q, dt, t=None):
        """用产热 Q(每电芯 W) 推进 dt 秒。

        采用隐式欧拉(后向欧拉)求解热网络，对导热/对流**无条件稳定**，
        因此任意大的导热系数 G 或时间步 dt 都不会发散：

            (C_th - dt·M) · T_new = C_th·T_old + dt·(Q + h·T_amb)

        其中 M = K_导热 - diag(h)，K_导热 的对角线已为负和。

        时间语义：``t=None`` 时自动按 ``self._t += dt`` 推进（默认行为）；
        也可显式传入 ``t`` 覆盖。``self._t`` 会随推进更新，使 callable 型
        环境温度 ``T_amb(t)`` 在整段仿真中按真实时间取值（旧实现始终取 t=0）。
        """
        if t is None:
            t = self._t + dt
        Q = np.asarray(Q, dtype=float)
        Tamb = self.ambient(t)
        M = self.K - np.diag(self.h)  # 导热 + 对流 的线性算子
        A = np.diag(self.C_th) - dt * M
        rhs = self.C_th * self.T + dt * (Q + self.h * Tamb)
        self.T = np.linalg.solve(A, rhs)
        self._t = t
        return self.T.copy()

    def set_temperature(self, T):
        self.T = np.asarray(T, dtype=float)


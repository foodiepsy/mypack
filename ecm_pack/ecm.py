# ecm.py
# 可定制的等效电路（Thévenin）电芯模型
#
# 电路结构（默认 1 个 RC，可扩展到 N 个）：
#
#   OCV ──┬── R0 ──┬── [R1‖C1] ──┬── [R2‖C2] ── ... ──► V_terminal
#          │        │              │
#       (欧姆内阻)  v_rc1         v_rcN
#
# 若启用扩散(ECMD)：再叠加一个由分布 SoC 解算的扩散过电势 η_diff。
#
# 关键设计：RC 支路用「解析指数积分」推进，对时间步长无条件稳定，
# 因此无需刚性 ODE 求解器，整包仿真可固定大步长推进。
import numpy as np

from .data import as_callable


def _thomas(main, lower, upper, rhs):
    """Thomas 算法求解三对角方程组 A·x = rhs。
    main  : 对角线 (n,)
    lower : 下对角线 (n-1,)
    upper : 上对角线 (n-1,)
    """
    n = len(main)
    cp = np.zeros(n)  # 消元后的上对角线
    dp = np.zeros(n)  # 消元后的右端
    cp[0] = upper[0] / main[0]
    dp[0] = rhs[0] / main[0]
    for i in range(1, n):
        m = main[i] - lower[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class ECMCellSpec:
    """
    单个电芯的 ECM 规格（参数可定制）。

    参数
    ----
    capacity : float [A.h]
        电芯容量，SoC 演化的分母。
    ocv : 可调用 / 查表 / 标量
        OCV 作为 SoC 的函数：ocv(soc) -> V。
    R0 : 可调用 / 查表 / 标量
        欧姆内阻，作为 (T[degC], I[A], SoC) 的函数。
    R : list
        N 个 RC 支路的电阻，每个为 (T, I, SoC) -> Ohm 的函数/查表。
    C : list
        N 个 RC 支路的电容，每个为 (T, I, SoC) -> F 的函数/查表。
        dUdT : 可选, (T, SoC) -> V/K
            熵变系数，用于计算可逆热 Q_rev = -I·T·dUdT。None 表示忽略。
        R_contact : 标量或 (T,I,SoC) 可调用 [Ω]
            外部连接电阻（tab/焊接/busbar），附加到 R0 外侧。
            产热 Q_cr = I²·R_contact，端电压 Vt = V_behind_R0 - I·(R0+R_contact)。
            默认 0.0（无连接电阻）。
        soc_init, T_init : float
        初始荷电状态(0~1) 与初始温度(K)。
    diffusion : bool
        是否启用 ECMD 扩散过电势（分布 SoC 1D PDE）。
    tau_D : float [s]
        扩散时间常数（仅 diffusion=True 时生效）。
    nx : int
        扩散分布 SoC 的网格点数。
    """

    def __init__(
        self,
        capacity,
        ocv,
        R0,
        R=None,
        C=None,
        dUdT=None,
        R_contact=0.0,
        soc_init=1.0,
        T_init=298.15,
        diffusion=False,
        tau_D=100.0,
        nx=12,
        # 三维热模型几何参数（用于 Cell3DThermal）
        Lx=None, Ly=None, Lz=None,
        rho=None, cp=None, k=None,
    ):
        self.capacity = float(capacity)
        self.ocv = as_callable(ocv)
        self.R0 = as_callable(R0)
        R = R or []
        C = C or []
        if len(R) != len(C):
            raise ValueError("R 与 C 的 RC 支路数量必须一致")
        self.R = [as_callable(r) for r in R]
        self.C = [as_callable(c) for c in C]
        self.dUdT = as_callable(dUdT) if dUdT is not None else None
        self.R_contact = as_callable(R_contact)
        self.n_rc = len(R)
        self.soc_init = float(soc_init)
        self.T_init = float(T_init)
        self.diffusion = bool(diffusion)
        self.tau_D = float(tau_D)
        self.nx = int(nx)
        # 三维热模型几何参数（可选；填入后可驱动 Cell3DThermal）
        self.Lx = float(Lx) if Lx is not None else None
        self.Ly = float(Ly) if Ly is not None else None
        self.Lz = float(Lz) if Lz is not None else None
        self.rho = float(rho) if rho is not None else None
        self.cp = float(cp) if cp is not None else None
        self.k = k  # 标量或三元组，原样保存

    def clone(self, **overrides):
        """复制规格并覆盖部分字段（用于构造参数略有差异的整包电芯）。"""
        kwargs = dict(
            capacity=self.capacity,
            ocv=self.ocv,
            R0=self.R0,
            R=self.R,
            C=self.C,
            dUdT=self.dUdT,
            R_contact=self.R_contact,
            soc_init=self.soc_init,
            T_init=self.T_init,
            diffusion=self.diffusion,
            tau_D=self.tau_D,
            nx=self.nx,
            Lx=self.Lx, Ly=self.Ly, Lz=self.Lz,
            rho=self.rho, cp=self.cp, k=self.k,
        )
        kwargs.update(overrides)
        return ECMCellSpec(**kwargs)


class ECMCell:
    """持有时序状态的 ECM 电芯。"""

    def __init__(self, spec):
        self.spec = spec
        self.reset()

    def reset(self):
        self.soc = self.spec.soc_init
        self.T = self.spec.T_init
        self.v_rc = np.zeros(self.spec.n_rc)
        if self.spec.diffusion:
            self.z = np.ones(self.spec.nx) * self.soc

    # ---------- 电压量 ----------
    def ocv(self):
        return self.spec.ocv(self.soc)

    def eta_diffusion(self):
        if not self.spec.diffusion:
            return 0.0
        z_surf = self.z[-1]
        return -(self.spec.ocv(z_surf) - self.spec.ocv(self.soc))

    def voltage_behind_R0(self):
        """R0 之后的等效电动势 E = OCV + Σv_rc + η_diff。
        网表中把它作为电压源值，再串一个 R0 电阻。"""
        return self.ocv() + float(self.v_rc.sum()) + self.eta_diffusion()

    # ---------- 电学步进 ----------
    def step_electrical(self, I, dt):
        """
        用电流 I 推进一个时间步 dt。返回本步的欧姆内阻 R0（供网表使用）。
        RC 支路采用解析解：v(t+dt) = v_inf + (v(0)-v_inf)·exp(-dt/τ)，
        其中稳态 v_inf = -I·R，对 dt 无条件稳定。
        """
        Td = self.T - 273.15
        R0 = self.spec.R0(Td, I, self.soc)
        Rs = np.array([r(Td, I, self.soc) for r in self.spec.R], dtype=float)
        Cs = np.array([c(Td, I, self.soc) for c in self.spec.C], dtype=float)
        tau = np.where(Rs * Cs > 0, Rs * Cs, 1e9)
        for k in range(self.spec.n_rc):
            v_inf = -I * Rs[k]
            self.v_rc[k] = v_inf + (self.v_rc[k] - v_inf) * np.exp(-dt / tau[k])
        # SoC：线性 ODE 的精确推进
        self.soc += -I * dt / (self.spec.capacity * 3600.0)
        self.soc = min(max(self.soc, 0.0), 1.0)
        if self.spec.diffusion:
            self._step_diffusion(I, dt)
        return R0

    def _step_diffusion(self, I, dt):
        """分布 SoC 的 1D 扩散：隐式欧拉(后向欧拉) + Thomas 三对角求解，
        任意 dt 都稳定。边界：左 Neumann=0，右通量 = -tau_D·I/(Q·3600)。"""
        n = self.spec.nx
        dx = 1.0 / (n - 1)
        tau_D = self.spec.tau_D
        z = self.z
        r = dt / (tau_D * dx * dx)  # 扩散数（隐式下无稳定性限制）
        # 构造 (I - r·L) z_new = z_old
        # 内部点：z_i - r(z_{i+1}-2z_i+z_{i-1}) = z_old_i
        main = np.ones(n) * (1.0 + 2.0 * r)
        lower = np.ones(n - 1) * (-r)
        upper = np.ones(n - 1) * (-r)
        # Neumann 边界（两端）：虚节点法使 Laplacian 在边界退化为 2(邻居-自身)，
        # 因此边界行的次对角系数必须为 **-2r**（非 -r），否则分布 SoC 质量不守恒。
        if n > 1:
            lower[-1] = -2.0 * r   # 右边界行：系数作用在 z_{n-2}
            upper[0] = -2.0 * r    # 左边界行：系数作用在 z_1
        # 右边界通量：用虚节点 z_{n} = z_{n-2} + 2·dx·J，J = -tau_D·I/(Q·3600)
        J = -tau_D * I / (self.spec.capacity * 3600.0)
        # 右端项修正：把通量并入最后一个方程
        rhs = z.copy()
        rhs[-1] = z[-1] + 2.0 * r * dx * J
        z_new = _thomas(main, lower, upper, rhs)
        self.z = z_new

    # ---------- 热 ----------
    def heat(self, I, R0):
        """产热：Q = I²R0 + Σ(-I·v_rc) + Q_rev（可逆熵热）。单位 W。"""
        Q = I**2 * R0 + float(np.sum(-I * self.v_rc))
        if self.spec.dUdT is not None:
            Td = self.T - 273.15
            Q += -I * self.T * self.spec.dUdT(Td, self.soc)
        return Q

    def terminal_voltage(self, R0, I):
        return self.voltage_behind_R0() - I * R0

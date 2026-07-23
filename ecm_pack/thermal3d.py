# thermal3d.py
# 三维电芯热模型：支持填入电芯长宽高，内部用有限体积法(FVM)离散化。
#
# 物理模型：
#   ρ·cp · ∂T/∂t = ∇·(k·∇T) + q_vol
#
# 其中 q_vol [W/m³] 为电芯内部体积产热率（来自 ECM 的 heat()/体积）。
# 边界条件：电芯六个面与外界对流换热 h [W/(m²·K)]，环境温度 T_amb。
#
# 数值方法：
#   - 结构化均匀网格 (nx × ny × nz)，有限体积法离散；
#   - 隐式欧拉时间推进，对时间步长无条件稳定；
#   - 三维扩散用 7 点拉普拉斯算子（中心 + 6 邻居）；
#   - 大型稀疏线性方程组用 scipy.sparse 求解。
#
# 这是对 PyBaMM "双集总(电芯+夹具)" 思想的升级：直接在三维空间求解
# 温度场，可获得电芯内部温度分布（热点、温度梯度），满足大容量电芯
# （如 314Ah）对热设计的工程需求。
import numpy as np
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import spsolve


class Cell3DThermal:
    """
    单个电芯的三维热模型。

    参数
    ----
    Lx, Ly, Lz : float [m]
        电芯几何尺寸（长 × 宽 × 高）。
    nx, ny, nz : int
        三个方向的网格点数。
    rho : float [kg/m³]
        电芯平均密度。
    cp : float [J/(kg·K)]
        电芯平均比热容。
    k : float 或 (kx,ky,kz) [W/(m·K)]
        导热系数。标量=各向同性；三元组=各向异性（叠层电芯常 k_z << k_x,y）。
    h : float [W/(m²·K)]
        六个面对环境的对流换热系数。
    T_amb : float 或 callable(t)->K
        环境温度，可随时间变化。
    T_init : float [K]
        初始温度（均匀）。
    """

    def __init__(
        self,
        Lx, Ly, Lz,
        nx=6, ny=6, nz=10,
        rho=2500.0, cp=1100.0, k=1.5,
        h=5.0, T_amb=298.15, T_init=298.15,
    ):
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.rho = float(rho)
        self.cp = float(cp)
        if np.isscalar(k):
            self.kx = self.ky = self.kz = float(k)
        else:
            self.kx, self.ky, self.kz = (float(v) for v in k)
        self.h = float(h)
        self.T_amb = T_amb
        self.T_init = float(T_init)

        self.nx, self.ny, self.nz = max(self.nx, 2), max(self.ny, 2), max(self.nz, 2)
        self.N = self.nx * self.ny * self.nz  # 总网格数
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.dz = self.Lz / (self.nz - 1)
        self.volume = self.Lx * self.Ly * self.Lz

        # 温度场（展平为长度 N 的一维数组，索引 = i*ny*nz + j*nz + k）
        self.T = np.full(self.N, self.T_init)
        self._t = 0.0
        self._A = None  # 延迟构建矩阵（首次 step 时构建）

    def _idx(self, i, j, k):
        return i * self.ny * self.nz + j * self.nz + k

    def _build_matrix(self, dt):
        """构建隐式欧拉矩阵 A = M - dt·K，其中 M=ρ·cp·V_cell，K=扩散+对流算子。"""
        nx, ny, nz = self.nx, self.ny, self.nz
        N = self.N
        dx, dy, dz = self.dx, self.dy, self.dz
        kx, ky, kz = self.kx, self.ky, self.kz
        h = self.h

        # 单元体积与面面积
        V = dx * dy * dz  # 均匀网格
        Ax = dy * dz
        Ay = dx * dz
        Az = dx * dy

        rows, cols, vals = [], [], []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n = self._idx(i, j, k)
                    # 扩散：与 6 个邻居的导热
                    neighbors = []
                    if i > 0: neighbors.append((self._idx(i-1, j, k), kx * Ax / dx))
                    if i < nx-1: neighbors.append((self._idx(i+1, j, k), kx * Ax / dx))
                    if j > 0: neighbors.append((self._idx(i, j-1, k), ky * Ay / dy))
                    if j < ny-1: neighbors.append((self._idx(i, j+1, k), ky * Ay / dy))
                    if k > 0: neighbors.append((self._idx(i, j, k-1), kz * Az / dz))
                    if k < nz-1: neighbors.append((self._idx(i, j, k+1), kz * Az / dz))

                    diag_val = 0.0
                    for nb, g in neighbors:
                        rows.append(n); cols.append(nb); vals.append(-dt * g)  # 非对角为负
                        diag_val += dt * g
                    # 对流边界（暴露的面）
                    if i == 0: diag_val += dt * h * Ax
                    if i == nx-1: diag_val += dt * h * Ax
                    if j == 0: diag_val += dt * h * Ay
                    if j == ny-1: diag_val += dt * h * Ay
                    if k == 0: diag_val += dt * h * Az
                    if k == nz-1: diag_val += dt * h * Az

                    # 对角线 = +(ρ·cp·V + 扩散 + 对流)，正定对角占优
                    rho_cp_V = self.rho * self.cp * V
                    rows.append(n); cols.append(n); vals.append(rho_cp_V + diag_val)

        A = csr_matrix((vals, (rows, cols)), shape=(N, N))
        self._A = A
        self._A_dt = dt
        self._V_cell = V
        return A

    def step(self, Q_total, dt, t=None):
        """
        用电芯总产热 Q_total [W] 推进一个时间步 dt [s]。

        把总产热均匀分配到所有网格（体积产热率 q_vol = Q_total / 体积），
        然后求解三维热传导方程。

        返回更新后的温度场（展平数组，长度 N）。
        """
        if t is None:
            t = self._t + dt
        Tamb = self.T_amb(t) if callable(self.T_amb) else float(self.T_amb)

        # 体积产热率 [W/m³]
        q_vol = Q_total / self.volume if self.volume > 0 else 0.0

        if self._A is None or abs(self._A_dt - dt) > 1e-12:
            self._build_matrix(dt)

        V = self._V_cell
        rhs = self.rho * self.cp * V * self.T + dt * q_vol * V
        # 对流边界贡献到右端
        # (已包含在矩阵 A 的对角线中，这里 rhs 需要加 h·A_face·T_amb)
        # 由于矩阵中边界对流已加到对角线（减去），rhs 需要补上 h·A·T_amb
        nx, ny, nz = self.nx, self.ny, self.nz
        Ax = self.dy * self.dz; Ay = self.dx * self.dz; Az = self.dx * self.dy
        h = self.h
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n = self._idx(i, j, k)
                    bnd = 0.0
                    if i == 0: bnd += h * Ax
                    if i == nx-1: bnd += h * Ax
                    if j == 0: bnd += h * Ay
                    if j == ny-1: bnd += h * Ay
                    if k == 0: bnd += h * Az
                    if k == nz-1: bnd += h * Az
                    rhs[n] += dt * bnd * Tamb

        self.T = spsolve(self._A, rhs)
        self._t = t
        return self.T

    def temperature_stats(self):
        """返回温度场统计：最高、最低、平均、最大温差。"""
        return {
            "T_max [K]": float(self.T.max()),
            "T_min [K]": float(self.T.min()),
            "T_avg [K]": float(self.T.mean()),
            "dT_max [K]": float(self.T.max() - self.T.min()),
        }

    @property
    def T_avg(self):
        return float(self.T.mean())

    @property
    def T_max(self):
        return float(self.T.max())

    def reshape(self):
        """返回三维温度场数组 (nx, ny, nz)。"""
        return self.T.reshape(self.nx, self.ny, self.nz)

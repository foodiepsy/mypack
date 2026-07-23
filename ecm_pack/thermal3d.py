# thermal3d.py
# 电芯多维热模型：支持 1D / 2D / 3D 切换，填入电芯长宽高与各向异性热物性。
#
# 物理模型：
#   ρ·cp · ∂T/∂t = ∇·(k·∇T) + q_vol
#
# 其中 q_vol [W/m³] 为体积产热率（来自 ECM 的 heat()/体积）。
# 边界条件：所有暴露面与外界对流换热 h [W/(m²·K)]，环境温度 T_amb。
#
# 维度切换：
#   dim=1 : 仅沿 X(宽度) 方向求解，Y/Z 方向绝热（侧面无限大近似）
#   dim=2 : 在 X-Y 平面求解，Z 方向绝热
#   dim=3 : 三个方向完整求解
#
# 数值方法：
#   - 结构化均匀网格，有限体积法(FVM)离散；
#   - 隐式欧拉时间推进，对时间步长无条件稳定；
#   - 用 scipy.sparse 求解大型稀疏线性方程组。
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class CellThermalModel:
    """
    单个电芯的多维热模型（1D/2D/3D 可切换）。

    参数
    ----
    Lx, Ly, Lz : float [m]
        电芯几何尺寸。约定：X=宽度, Y=厚度, Z=高度。
    dim : int (1/2/3)
        求解维度。1=仅X方向, 2=XY平面, 3=完整三维。
    nx, ny, nz : int
        各方向网格点数（低于 dim 的方向自动设为 1）。
    rho : float [kg/m³]
        电芯平均密度。
    cp : float [J/(kg·K)]
        电芯平均比热容。
    k : float 或 (kx,ky,kz) [W/(m·K)]
        导热系数。标量=各向同性；三元组=各向异性。
        约定：kx=X(宽度)方向, ky=Y(厚度)方向, kz=Z(高度)方向。
    h : float [W/(m²·K)]
        暴露面对环境的对流换热系数。
    T_amb : float 或 callable(t)->K
        环境温度，可随时间变化。
    T_init : float [K]
        初始温度（均匀）。

    属性
    ----
    T : ndarray
        温度场（展平为一维数组，长度 = 活跃网格点数）。
    """

    def __init__(
        self,
        Lx, Ly, Lz,
        dim=3,
        nx=6, ny=6, nz=10,
        rho=2300.0, cp=1000.0, k=1.5,
        h=5.0, T_amb=298.15, T_init=298.15,
    ):
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.dim = int(dim)
        if self.dim not in (1, 2, 3):
            raise ValueError("dim 必须为 1、2 或 3")
        self.rho = float(rho)
        self.cp = float(cp)
        if np.isscalar(k):
            self.kx = self.ky = self.kz = float(k)
        else:
            self.kx, self.ky, self.kz = (float(v) for v in k)
        self.h = float(h)
        self.T_amb = T_amb
        self.T_init = float(T_init)

        # 按 dim 截断网格：低于 dim 的方向只取 1 个节点
        self.nx = max(int(nx), 2) if self.dim >= 1 else 1
        self.ny = max(int(ny), 2) if self.dim >= 2 else 1
        self.nz = max(int(nz), 2) if self.dim >= 3 else 1

        self.N = self.nx * self.ny * self.nz
        self.dx = self.Lx / (self.nx - 1) if self.nx > 1 else self.Lx
        self.dy = self.Ly / (self.ny - 1) if self.ny > 1 else self.Ly
        self.dz = self.Lz / (self.nz - 1) if self.nz > 1 else self.Lz
        self.volume = self.Lx * self.Ly * self.Lz

        self.T = np.full(self.N, self.T_init)
        self._t = 0.0
        self._A = None
        self._A_dt = None
        self._V_cell = None
        self._boundary_cache = None  # 缓存边界节点的对流面积

    def _idx(self, i, j, k):
        return i * self.ny * self.nz + j * self.nz + k

    def _is_boundary(self, i, j, k):
        """返回该节点的暴露面总对流面积 [m²]。"""
        Ax = self.dy * self.dz
        Ay = self.dx * self.dz
        Az = self.dx * self.dy
        area = 0.0
        if self.nx > 1:
            if i == 0 or i == self.nx - 1:
                area += Ax
        if self.ny > 1:
            if j == 0 or j == self.ny - 1:
                area += Ay
        if self.nz > 1:
            if k == 0 or k == self.nz - 1:
                area += Az
        return area

    def _build_matrix(self, dt):
        """构建隐式欧拉矩阵 A（正定对角占优）与边界面积缓存。"""
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        kx, ky, kz = self.kx, self.ky, self.kz

        V = dx * dy * dz
        Ax = dy * dz
        Ay = dx * dz
        Az = dx * dy

        rows, cols, vals = [], [], []
        boundary_areas = np.zeros(self.N)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n = self._idx(i, j, k)
                    diag_val = 0.0

                    # X 方向邻居
                    if nx > 1:
                        if i > 0:
                            g = kx * Ax / dx
                            rows.append(n); cols.append(self._idx(i-1, j, k))
                            vals.append(-dt * g); diag_val += dt * g
                        if i < nx - 1:
                            g = kx * Ax / dx
                            rows.append(n); cols.append(self._idx(i+1, j, k))
                            vals.append(-dt * g); diag_val += dt * g
                    # Y 方向邻居
                    if ny > 1:
                        if j > 0:
                            g = ky * Ay / dy
                            rows.append(n); cols.append(self._idx(i, j-1, k))
                            vals.append(-dt * g); diag_val += dt * g
                        if j < ny - 1:
                            g = ky * Ay / dy
                            rows.append(n); cols.append(self._idx(i, j+1, k))
                            vals.append(-dt * g); diag_val += dt * g
                    # Z 方向邻居
                    if nz > 1:
                        if k > 0:
                            g = kz * Az / dz
                            rows.append(n); cols.append(self._idx(i, j, k-1))
                            vals.append(-dt * g); diag_val += dt * g
                        if k < nz - 1:
                            g = kz * Az / dz
                            rows.append(n); cols.append(self._idx(i, j, k+1))
                            vals.append(-dt * g); diag_val += dt * g

                    # 对流边界
                    bnd_area = self._is_boundary(i, j, k)
                    diag_val += dt * self.h * bnd_area
                    boundary_areas[n] = bnd_area

                    rho_cp_V = self.rho * self.cp * V
                    rows.append(n); cols.append(n)
                    vals.append(rho_cp_V + diag_val)

        self._A = csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        self._A_dt = dt
        self._V_cell = V
        self._boundary_cache = boundary_areas
        return self._A

    def step(self, Q_total, dt, t=None):
        """
        用电芯总产热 Q_total [W] 推进一个时间步 dt [s]。
        产热均匀分配到所有网格体积。返回更新后的温度场（展平数组）。
        """
        if t is None:
            t = self._t + dt
        Tamb = self.T_amb(t) if callable(self.T_amb) else float(self.T_amb)

        q_vol = Q_total / self.volume if self.volume > 0 else 0.0

        if self._A is None or self._A_dt is None or abs(self._A_dt - dt) > 1e-12:
            self._build_matrix(dt)

        V = self._V_cell
        rho_cp_V = self.rho * self.cp * V
        rhs = rho_cp_V * self.T + dt * q_vol * V
        # 对流边界贡献：+ dt·h·A_bnd·T_amb
        rhs += dt * self.h * self._boundary_cache * Tamb

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
        """返回温度场数组，形状按 dim 对应 (nx,)/ (nx,ny)/ (nx,ny,nz)。"""
        if self.dim == 1:
            return self.T.copy()
        elif self.dim == 2:
            return self.T.reshape(self.nx, self.ny)
        else:
            return self.T.reshape(self.nx, self.ny, self.nz)


# 向后兼容别名
Cell3DThermal = CellThermalModel

# thermal3d.py
# 电芯多维热模型：支持 1D / 2D / 3D 切换，填入电芯长宽高与各向异性热物性。

# 物理模型：
# ρ·cp · ∂T/∂t = ∇·(k·∇T) + q_vol

# 其中 q_vol [W/m³] 为体积产热率（来自 ECM 的 heat()/体积）。
# 边界条件：所有暴露面与外界对流换热 h [W/(m²·K)]，环境温度 T_amb。

# 维度切换：
# dim=1 : 仅沿 X(宽度) 方向求解，Y/Z 方向绝热（侧面无限大近似）
# dim=2 : 在 X-Y 平面求解，Z 方向绝热
# dim=3 : 三个方向完整求解

# 数值方法：
# - 结构化均匀网格，有限体积法(FVM)离散；
# - 隐式欧拉时间推进，对时间步长无条件稳定；
# - 用 scipy.sparse 求解大型稀疏线性方程组。
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
class CellThermalModel:
    """
    单个电芯的多维热模型（1D/2D/3D 可切换，支持非对称冷却）。
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
    h : float / 6-tuple / dict [W/(m²·K)]
        各面对环境的对流换热系数。支持三种格式：
          - 标量: 6 个面统一（向后兼容）
          - 6-元组: (h_x0,h_x1,h_y0,h_y1,h_z0,h_z1)
          - dict: {'x0':20,'x1':5, ...} 或带 'default' 键回退
        面命名约定:
          x0 = X=0 面(i==0)  x1 = X=Lx 面(i==nx-1)
          y0 = Y=0 面(j==0)  y1 = Y=Ly 面(j==ny-1)
          z0 = Z=0 面(k==0)  z1 = Z=Lz 面(k==nz-1)
        典型场景: 水冷板在 x0 面 (h=200),其余自然对流 (h=5)
    T_amb : float 或 callable(t)->K
        环境温度，可随时间变化。
    T_init : float [K]
        初始温度（均匀）。
    R_shell : float [K/W], 可选
        电芯壳层热阻（体→表面的热阻）。>0 时在 FVM 解之后额外计算表面温度
        T_surface 供 BMS 传感器对标。默认 0.0（体温度即为表面温度）。

    属性
    ----
    T : ndarray
        温度场（展平为一维数组，长度 = 活跃网格点数）。
    h_faces : tuple (hx0,hx1,hy0,hy1,hz0,hz1)
        各面对流系数。
    T_surface : float  [K]
        壳层/表面温度（当 R_shell>0 时，为从体平均温度反推的表面温度）。
    T_core_max : float  [K]
        电芯体最高温度（= T_max，同样从 FVM 场获取）。
    """

    def __init__(
        self,
        Lx, Ly, Lz,
        dim=3,
        nx=6, ny=6, nz=10,
        rho=2300.0, cp=1000.0, k=1.5,
        h=5.0, T_amb=298.15, T_init=298.15,
        R_shell=0.0,
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
        self._h_faces = self._parse_h(h)
        self.T_amb = T_amb
        self.T_init = float(T_init)
        self.R_shell = max(0.0, float(R_shell))

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
        self._boundary_hA_cache = None  # 缓存每节点的 Σ(h_face·A_face)
        self._h_total = 0.0              # 总对流导 [W/K] = Σ(h_face·A_face)
        self._T_surface = float(T_init)  # 壳层温度（R_shell>0 时有意义）

    @staticmethod
    def _parse_h(h):
        """将 h 统一解析为 (h_x0,h_x1,h_y0,h_y1,h_z0,h_z1) 六元组。"""
        if isinstance(h, (int, float, np.number)):
            h = float(h)
            return (h, h, h, h, h, h)
        if isinstance(h, str):
            raise TypeError(f"h 参数不能为字符串，收到: {repr(h)}")
        if isinstance(h, dict):
            default = float(h.get("default", 5.0))
            return tuple(float(h.get(k, default))
                         for k in ("x0", "x1", "y0", "y1", "z0", "z1"))
        if isinstance(h, (list, tuple)):
            if len(h) == 6:
                return tuple(float(v) for v in h)
            raise ValueError(f"h 元组必须为 6 元素，收到 {len(h)}")
        raise TypeError(f"h 参数类型不支持: {type(h)}")

    @property
    def h_faces(self):
        """返回各面对流系数的六元组 (hx0,hx1,hy0,hy1,hz0,hz1)。"""
        return self._h_faces

    def _idx(self, i, j, k):
        return i * self.ny * self.nz + j * self.nz + k

    def _build_matrix(self, dt):
        """构建隐式欧拉矩阵 A，缓存每节点的 Σ(h_face·A_face)。"""
        nx, ny, nz = self.nx, self.ny, self.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        kx, ky, kz = self.kx, self.ky, self.kz
        hx0, hx1, hy0, hy1, hz0, hz1 = self._h_faces

        V_full = dx * dy * dz                  # 单元(全)控制体体积
        Ax = dy * dz
        Ay = dx * dz
        Az = dx * dy

        rows, cols, vals = [], [], []
        boundary_hA = np.zeros(self.N)         # 缓存 hA = Σ(h_face·area_face)
        V_node = np.empty(self.N)              # 每节点控制体体积(边界减半, 见下)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n = self._idx(i, j, k)
                    diag_val = 0.0

                    # 节点控制体边界因子：内部=1，边界=½；退化维(该方向仅 1 节点)
                    # 节点独占整段→取 1。同步用于“控制体体积”和“控制体各面面积”，
                    # 以符合顶点中心 FVM 几何：边界节点的控制体在对应方向只占半格，
                    # 其切向面也须按因子折半，否则 hA/V、g 偏大(边/角尤甚)、几何不一致(S1)。
                    fx = 0.5 if (nx > 1 and (i == 0 or i == nx - 1)) else 1.0
                    fy = 0.5 if (ny > 1 and (j == 0 or j == ny - 1)) else 1.0
                    fz = 0.5 if (nz > 1 and (k == 0 or k == nz - 1)) else 1.0

                    # X 方向邻居（传导面面积随切向因子缩放；对称配对，矩阵保持对称）
                    if nx > 1:
                        if i > 0:
                            g = kx * Ax / dx * fy * fz
                            rows.append(n); cols.append(self._idx(i-1, j, k))
                            vals.append(-dt * g); diag_val += dt * g
                        if i < nx - 1:
                            g = kx * Ax / dx * fy * fz
                            rows.append(n); cols.append(self._idx(i+1, j, k))
                            vals.append(-dt * g); diag_val += dt * g
                    # Y 方向邻居
                    if ny > 1:
                        if j > 0:
                            g = ky * Ay / dy * fx * fz
                            rows.append(n); cols.append(self._idx(i, j-1, k))
                            vals.append(-dt * g); diag_val += dt * g
                        if j < ny - 1:
                            g = ky * Ay / dy * fx * fz
                            rows.append(n); cols.append(self._idx(i, j+1, k))
                            vals.append(-dt * g); diag_val += dt * g
                    # Z 方向邻居
                    if nz > 1:
                        if k > 0:
                            g = kz * Az / dz * fx * fy
                            rows.append(n); cols.append(self._idx(i, j, k-1))
                            vals.append(-dt * g); diag_val += dt * g
                        if k < nz - 1:
                            g = kz * Az / dz * fx * fy
                            rows.append(n); cols.append(self._idx(i, j, k+1))
                            vals.append(-dt * g); diag_val += dt * g

                    # ─── 对流边界（面差异化 h）───
                    # 对流面面积同样随切向边界因子缩放；退化维(if nd>1 守卫)按设计绝热。
                    hA_node = 0.0
                    if nx > 1:
                        if i == 0:
                            hA_node += hx0 * Ax * fy * fz
                        if i == nx - 1:
                            hA_node += hx1 * Ax * fy * fz
                    if ny > 1:
                        if j == 0:
                            hA_node += hy0 * Ay * fx * fz
                        if j == ny - 1:
                            hA_node += hy1 * Ay * fx * fz
                    if nz > 1:
                        if k == 0:
                            hA_node += hz0 * Az * fx * fy
                        if k == nz - 1:
                            hA_node += hz1 * Az * fx * fy
                    diag_val += dt * hA_node
                    boundary_hA[n] = hA_node

                    # 节点控制体体积：内部全体积，边界减半（面½/棱¼/角⅛）。
                    # ΣV_node == 真实体积，恢复能量守恒，并消除粗网格温度高估(S1)。
                    vn = V_full * fx * fy * fz
                    V_node[n] = vn
                    rho_cp_V = self.rho * self.cp * vn
                    rows.append(n); cols.append(n)
                    vals.append(rho_cp_V + diag_val)

        self._A = csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        self._A_dt = dt
        self._V_cell = V_node
        self._boundary_hA_cache = boundary_hA
        # 缓存总对流导 h_total [W/K]：每面 = h_face × 物理面积
        h_total = 0.0
        if nx > 1:
            h_total += (hx0 + hx1) * self.Ly * self.Lz
        if ny > 1:
            h_total += (hy0 + hy1) * self.Lx * self.Lz
        if nz > 1:
            h_total += (hz0 + hz1) * self.Lx * self.Ly
        self._h_total = h_total
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
        # 对流边界贡献（面差异化 h）：+ dt·Σ(h_face·A_face)·T_amb
        rhs += dt * self._boundary_hA_cache * Tamb

        self.T = spsolve(self._A, rhs)
        self._t = t

        # ─── 壳层表面温度（R_shell>0 时计算）───
        if self.R_shell > 0 and self._h_total > 0:
            T_avg = self._T_bulk()
            inv_R = 1.0 / self.R_shell
            self._T_surface = (T_avg * inv_R + self._h_total * Tamb) / (inv_R + self._h_total)
        else:
            self._T_surface = self._T_bulk()
        return self.T

    def _T_bulk(self):
        """体积加权平均温度。

        修复后 _V_cell 为每节点非均匀控制体体积，算术平均会错算“体平均”
        （边界节点控制体小却被同等加权），对 R_shell 表面温度子模型引入偏差。
        未构建矩阵(_V_cell 为 None)时回退到算术平均。
        """
        V = self._V_cell
        if V is None:
            return float(self.T.mean())
        return float(np.average(self.T, weights=V))

    def temperature_stats(self):
        """返回温度场统计：最高、最低、平均、最大温差，及表面温度和体心最高温。"""
        stats = {
            "T_max [K]": float(self.T.max()),
            "T_min [K]": float(self.T.min()),
            "T_avg [K]": self._T_bulk(),
            "dT_max [K]": float(self.T.max() - self.T.min()),
        }
        if self.R_shell > 0:
            stats["T_surface [K]"] = self._T_surface
            stats["T_core_max [K]"] = float(self.T.max())
        return stats

    @property
    def T_avg(self):
        return self._T_bulk()

    @property
    def T_max(self):
        return float(self.T.max())

    @property
    def T_surface(self):
        """壳层表面温度 [K]。R_shell>0 时电芯内部体平均温度与外壳之差由 R_shell 决定。"""
        return self._T_surface

    @property
    def T_core_max(self):
        """电芯内部最高温度 [K]（即体温度场的最大值）。"""
        return float(self.T.max())

    @property
    def h_total(self):
        """总对流导 [W/K] = Σ(h_face × A_face)。"""
        return self._h_total

    def reshape(self):
        """返回温度场数组，形状按 dim 对应 (nx,)/ (nx,ny)/ (nx,ny,nz)。"""
        if self.dim == 1:
            return self.T.copy()
        elif self.dim == 2:
            return self.T.reshape(self.nx, self.ny)
        else:
            return self.T.reshape(self.nx, self.ny, self.nz)

    # ─────────── 热场可视化 ───────────

    def plot_slice(self, plane="xy", position=0.5, ax=None,
                   cmap="hot", show_colorbar=True, **kwargs):
        """绘制温度场的指定截面 heatmap。

        参数
        ----
        plane : 'xy' | 'xz' | 'yz'
            截面平面。1D 模型忽略此参数（画 1D 曲线）。
        position : float (0~1) 或 int
            截面位置（分数或索引）。默认 0.5（中截面）。
        ax : matplotlib Axes, 可选
            若提供则绘制到该轴。
        cmap : str
            色图名称，默认 'hot'。
        show_colorbar : bool
            是否显示色标。

        返回 fig, ax。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("plot_slice 需要 matplotlib") from e

        if self.dim == 1:
            # 1D：画温度随 X 的曲线
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 3))
            x_vals = np.linspace(0, self.Lx * 1000, self.nx)  # mm
            ax.plot(x_vals, self.T, "o-", color="darkred", lw=1.5, ms=4, **kwargs)
            ax.set(xlabel="X 位置 [mm]", ylabel="温度 [K]",
                   title=f"1D 温度分布 (t={self._t:.1f}s)")
            ax.grid(alpha=0.3)
            return ax.figure, ax

        T3d = self.reshape()
        nd = T3d.ndim if hasattr(T3d, "ndim") else \
            (1 if self.dim == 1 else (2 if self.dim == 2 else 3))

        # 确定切片索引
        if plane not in ("xy", "xz", "yz"):
            raise ValueError(f"plane 必须为 'xy'/'xz'/'yz'，收到 {plane!r}")

        if plane == "xy":
            if nd < 3 or self.nz <= 1:
                slc = T3d
                extent = (0, self.Lx * 1000, 0, self.Ly * 1000)
                xy_label = ("X [mm]", "Y [mm]")
                title = f"温度场 (t={self._t:.1f}s)"
            else:
                idx = _slice_index(self.nz, position)
                slc = T3d[:, :, idx]
                pos_mm = idx / (self.nz - 1) * self.Lz * 1000
                extent = (0, self.Lx * 1000, 0, self.Ly * 1000)
                xy_label = ("X [mm]", "Y [mm]")
                title = f"T @ Z={pos_mm:.1f}mm (t={self._t:.1f}s)"
        elif plane == "xz":
            if nd < 3 or self.ny <= 1:
                slc = T3d if nd == 2 else T3d[:, 0, :]
                extent = (0, self.Lx * 1000, 0, self.Lz * 1000)
                xy_label = ("X [mm]", "Z [mm]")
                title = f"温度场 (t={self._t:.1f}s)"
            else:
                idx = _slice_index(self.ny, position)
                slc = T3d[:, idx, :]
                pos_mm = idx / (self.ny - 1) * self.Ly * 1000
                extent = (0, self.Lx * 1000, 0, self.Lz * 1000)
                xy_label = ("X [mm]", "Z [mm]")
                title = f"T @ Y={pos_mm:.1f}mm (t={self._t:.1f}s)"
        else:  # yz
            if nd < 3 or self.nx <= 1:
                slc = T3d.T if nd == 2 else T3d[0, :, :]
                extent = (0, self.Ly * 1000, 0, self.Lz * 1000)
                xy_label = ("Y [mm]", "Z [mm]")
                title = f"温度场 (t={self._t:.1f}s)"
            else:
                idx = _slice_index(self.nx, position)
                slc = T3d[idx, :, :]
                pos_mm = idx / (self.nx - 1) * self.Lx * 1000
                extent = (0, self.Ly * 1000, 0, self.Lz * 1000)
                xy_label = ("Y [mm]", "Z [mm]")
                title = f"T @ X={pos_mm:.1f}mm (t={self._t:.1f}s)"

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(slc.T, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, **kwargs)
        ax.set(xlabel=xy_label[0], ylabel=xy_label[1])
        ax.set_title(title if "title" in dir() else f"温度场 (t={self._t:.1f}s)")

        if show_colorbar and hasattr(im, "figure") and im.figure is not None:
            plt.colorbar(im, ax=ax, label="温度 [K]")
        return ax.figure, ax

    def plot_summary(self, save_path=None, dpi=120):
        """生成热场概览图：1D/2D/3D 的可视化摘要。

        参数
        ----
        save_path : str, 可选
            保存路径。
        dpi : int

        返回 fig。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("plot_summary 需要 matplotlib") from e

        if self.dim == 1:
            fig, ax = plt.subplots(figsize=(7, 3))
            self.plot_slice(ax=ax)
            stats = self.temperature_stats()
            info = (f"T_avg={stats['T_avg [K]']:.2f}K  "
                    f"T_max={stats['T_max [K]']:.2f}K  "
                    f"ΔT={stats['dT_max [K]']:.3f}K")
            ax.set_title(f"1D 温度场 · {info}")
            fig.tight_layout()
        elif self.dim == 2:
            fig, ax = plt.subplots(figsize=(6, 5))
            self.plot_slice(plane="xy", ax=ax)
            stats = self.temperature_stats()
            info = (f"T_avg={stats['T_avg [K]']:.2f}K  "
                    f"T_max={stats['T_max [K]']:.2f}K  "
                    f"ΔT={stats['dT_max [K]']:.3f}K")
            ax.set_title(f"2D(XY)温度场 · {info}")
            fig.tight_layout()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for ax_i, plane in zip(axes, ("xy", "xz", "yz"), strict=True):
                self.plot_slice(plane=plane, ax=ax_i)
            stats = self.temperature_stats()
            extra = ""
            if "T_surface [K]" in stats:
                extra = (f"  T_surface={stats['T_surface [K]']:.2f}K"
                         f"  T_core_max={stats['T_core_max [K]']:.2f}K")
            fig.suptitle(
                f"三维温度场 (dim={self.dim}, t={self._t:.1f}s)  "
                f"T_avg={stats['T_avg [K]']:.2f}K  "
                f"ΔT_max={stats['dT_max [K]']:.3f}K{extra}",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.92])

        if save_path:
            fig.savefig(save_path, dpi=dpi)
        return fig

def _slice_index(n, pos):
    """辅助：将分数位置转为切片索引。"""
    if isinstance(pos, int):
        return max(0, min(pos, n - 1))
    return max(0, min(int(round(pos * (n - 1))), n - 1))
# 向后兼容别名
Cell3DThermal = CellThermalModel

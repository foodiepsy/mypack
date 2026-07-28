# thermal3d_stack.py
# 多电芯三维热模型（复合有限体积）：把若干电芯沿 Y(厚度) 轴大面背靠背直接堆叠，
# 泡棉只贴在电芯「薄侧」(侧向 X/Z 小面) 与空气连接，外边界逐面非对称对流冷却。
#
# 几何约定（按用户修正，2026-07-28）：
#   - 8 颗电芯沿 Y(厚度) 大面背靠背直接贴合，电芯之间「没有泡棉」；
#   - 泡棉只贴在特定薄侧面（由 foam_faces 指定，不一定是全部薄侧面），
#     无泡棉的薄侧面直接接空气对流（h_side、T_amb）；
#   - 顶部放塑料片（薄层，导热率 > 泡棉），再对流到 25°C 环境；
#   - 底部绝热（无换热）；
#   - 全局环境为 25°C 强制自然对流。
#
# 与 CellThermalModel 同一思路：
#   rho*cp * dT/dt = nabla·(k·nabla T) + q_vol
#   - 结构化均匀网格，有限体积法(FVM)离散；
#   - 隐式欧拉时间推进，对时间步长无条件稳定；
#   - scipy.sparse 稀疏矩阵 + 常量矩阵 LU 分解复用。
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


class StackThermal3D:
    """
    沿 Y(厚度) 轴堆叠 n_cells 颗电芯的三维热模型。
    电芯大面背靠背直接贴合（电芯间无泡棉）；泡棉只贴在指定薄侧；
    无泡棉的薄侧直接空气对流；顶部可配塑料薄层串联对流。

    参数
    ----
    n_cells : int
    Lx, Ly, Lz : float [m]           单颗电芯几何（约定 X=宽, Y=厚(堆叠), Z=高）
    nx, ny, nz : int                 单颗电芯网格点数（X,Y,Z 方向）
    cell_k : (kx,ky,kz) [W/(m·K)]    电芯各向异性导热
    cell_rho, cell_cp : float        电芯密度 / 比热容
    foam_k : float [W/(m·K)]          薄侧泡棉导热
    foam_thickness : float [m]       薄侧泡棉厚度（沿外法线方向）
    foam_faces : list[str], 可选     哪些薄侧面有泡棉，可选 "x0","x1","z0","z1"；
                                      默认["x0","x1","z0","z1"]=全部薄侧。
                                      不在 foam_faces 中的薄侧=直接空气对流。
    k_top : float, 可选              顶部塑料片导热率 [W/(m·K)]；0=无塑料,直接对流
    d_top : float, 可选              顶部塑料片厚度 [m]；0=无塑料
    h_top, T_top : float, 可选       顶部(y=0) 25°C强制对流系数与汇温度；0=绝热
    h_bottom, T_bottom : float, 可选  底部(y=max) 对流；0=绝热
    h_side, T_amb : float, 可选       薄侧空气强制对流系数与汇温度；0=绝热
    T_init : float, 可选             初始均匀温度 [K]
    """

    def __init__(
        self, n_cells, Lx, Ly, Lz, nx, ny, nz,
        cell_k, cell_rho, cell_cp,
        foam_k, foam_thickness,
        foam_faces=None,
        k_top=0.0, d_top=0.0,
        h_top=0.0, T_top=298.15,
        h_bottom=0.0, T_bottom=298.15,
        h_side=0.0, T_amb=298.15,
        T_init=298.15,
    ):
        self.n_cells = int(n_cells)
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.nx = max(int(nx), 1)
        self.nz = max(int(nz), 1)
        self.ny = max(int(ny), 1)

        if isinstance(cell_k, (int, float, np.number)):
            self.cell_kx = self.cell_ky = self.cell_kz = float(cell_k)
        else:
            self.cell_kx, self.cell_ky, self.cell_kz = (float(v) for v in cell_k)
        self.cell_rho = float(cell_rho)
        self.cell_cp = float(cell_cp)
        self.foam_k = float(foam_k)
        self.foam_thickness = float(foam_thickness)

        # 哪些薄侧贴泡棉（不在列表中的薄侧 = 直接空气对流）
        if foam_faces is None:
            self.foam_faces = {"x0", "x1", "z0", "z1"}
        else:
            self.foam_faces = set(foam_faces)

        self.k_top = float(k_top)
        self.d_top = float(d_top)

        self.h_top, self.T_top = float(h_top), float(T_top)
        self.h_bottom, self.T_bottom = float(h_bottom), float(T_bottom)
        self.h_side, self.T_amb = float(h_side), float(T_amb)
        self.T_init = float(T_init)

        # ---- 沿 Y 的块结构：电芯直接贴合，电芯间「无泡棉」----
        self._per_j = []
        dy_c = self.Ly / self.ny
        for c in range(self.n_cells):
            for _ in range(self.ny):
                self._per_j.append(dict(
                    ky=self.cell_ky, kx=self.cell_kx, kz=self.cell_kz,
                    rho=self.cell_rho, cp=self.cell_cp, dy=dy_c, cell=c))
        self.NY = len(self._per_j)
        self.NX = self.nx
        self.NZ = self.nz
        self.N = self.NX * self.NY * self.NZ

        self.dx = self.Lx / self.nx
        self.dz = self.Lz / self.nz
        self._V_cell = self.Lx * self.Ly * self.Lz

        self._vol = np.zeros(self.N)
        self._C = np.zeros(self.N)
        self._kx = np.zeros(self.N)
        self._ky = np.zeros(self.N)
        self._kz = np.zeros(self.N)
        self._dy = np.zeros(self.N)
        self._cell = np.full(self.N, -1, dtype=int)
        for j, pj in enumerate(self._per_j):
            for i in range(self.NX):
                for k in range(self.NZ):
                    n = self._idx(i, j, k)
                    dy = pj["dy"]
                    vol = self.dx * dy * self.dz
                    self._vol[n] = vol
                    self._C[n] = pj["rho"] * pj["cp"] * vol
                    self._kx[n] = pj["kx"]
                    self._ky[n] = pj["ky"]
                    self._kz[n] = pj["kz"]
                    self._dy[n] = dy
                    self._cell[n] = pj["cell"]
        self.T_field = np.full(self.N, self.T_init)
        self.T = np.full(self.n_cells, self.T_init)
        self._t = 0.0

        self._A = None
        self._lu = None
        self._dt = None
        self._bc_rhs = None

    def _idx(self, i, j, k):
        return i * self.NY * self.NZ + j * self.NZ + k

    def _in(self, i, j, k):
        return 0 <= i < self.NX and 0 <= j < self.NY and 0 <= k < self.NZ

    def reshape(self):
        return self.T_field.reshape(self.NX, self.NY, self.NZ)

    def _side_conductance(self, A_face):
        """薄侧泡棉+空气对流串联：g = A / (d_foam/k_foam + 1/h_side)"""
        if self.h_side <= 0:
            return 0.0
        return A_face / (self.foam_thickness / self.foam_k + 1.0 / self.h_side)

    def _top_conductance(self, Ay):
        """顶部塑料薄层+空气：g = Ay / (d_top/k_top + 1/h_top)；k_top=0 时直接对流"""
        if self.h_top <= 0:
            return 0.0
        if self.k_top > 0 and self.d_top > 0:
            return Ay / (self.d_top / self.k_top + 1.0 / self.h_top)
        return self.h_top * Ay

    def _build(self, dt):
        dx, dz = self.dx, self.dz
        NX, NY, NZ = self.NX, self.NY, self.NZ
        rows, cols, vals = [], [], []
        self._bc_rhs = np.zeros(self.N)

        foam_x0 = "x0" in self.foam_faces
        foam_x1 = "x1" in self.foam_faces
        foam_z0 = "z0" in self.foam_faces
        foam_z1 = "z1" in self.foam_faces

        for i in range(NX):
            for j in range(NY):
                pj = self._per_j[j]
                dy = pj["dy"]
                Ax = dy * dz
                Ay = dx * dz
                Az = dx * dy
                gx_foam = self._side_conductance(Ax)
                gx_bare = self.h_side * Ax if self.h_side > 0 else 0.0
                gz_foam = self._side_conductance(Az)
                gz_bare = self.h_side * Az if self.h_side > 0 else 0.0
                for k in range(NZ):
                    n = self._idx(i, j, k)
                    diag = 0.0

                    # ---- X 方向（块内）----
                    if NX > 1:
                        gx_in = self._kx[n] * Ax / dx
                        if i > 0:
                            nb = self._idx(i - 1, j, k)
                            rows.append(n); cols.append(nb); vals.append(-dt * gx_in)
                            diag += dt * gx_in
                        if i < NX - 1:
                            nb = self._idx(i + 1, j, k)
                            rows.append(n); cols.append(nb); vals.append(-dt * gx_in)
                            diag += dt * gx_in
                    # ---- Z 方向（块内）----
                    if NZ > 1:
                        gz_in = self._kz[n] * Az / dz
                        if k > 0:
                            nb = self._idx(i, j, k - 1)
                            rows.append(n); cols.append(nb); vals.append(-dt * gz_in)
                            diag += dt * gz_in
                        if k < NZ - 1:
                            nb = self._idx(i, j, k + 1)
                            rows.append(n); cols.append(nb); vals.append(-dt * gz_in)
                            diag += dt * gz_in
                    # ---- Y 方向（电芯直接贴合）----
                    if NY > 1:
                        if j > 0:
                            nb = self._idx(i, j - 1, k)
                            g = self._gy_conductance(n, nb, Ay)
                            rows.append(n); cols.append(nb); vals.append(-dt * g)
                            diag += dt * g
                        if j < NY - 1:
                            nb = self._idx(i, j + 1, k)
                            g = self._gy_conductance(n, nb, Ay)
                            rows.append(n); cols.append(nb); vals.append(-dt * g)
                            diag += dt * g

                    # ---- 顶部：塑料薄层 + 25°C 强制对流 (或直接对流) ----
                    if j == 0:
                        g_top = self._top_conductance(Ay)
                        if g_top > 0:
                            diag += dt * g_top
                            self._bc_rhs[n] += g_top * self.T_top
                    # ---- 底部：纯对流（或绝热）----
                    if j == NY - 1 and self.h_bottom > 0:
                        hA = self.h_bottom * Ay
                        diag += dt * hA
                        self._bc_rhs[n] += hA * self.T_bottom
                    # ---- 薄侧：贴泡棉的面 = 泡棉+空气串联；不贴泡棉的面 = 直接空气对流 ----
                    if i == 0:
                        g = gx_foam if foam_x0 else gx_bare
                        if g > 0:
                            diag += dt * g
                            self._bc_rhs[n] += g * self.T_amb
                    if i == NX - 1:
                        g = gx_foam if foam_x1 else gx_bare
                        if g > 0:
                            diag += dt * g
                            self._bc_rhs[n] += g * self.T_amb
                    if k == 0:
                        g = gz_foam if foam_z0 else gz_bare
                        if g > 0:
                            diag += dt * g
                            self._bc_rhs[n] += g * self.T_amb
                    if k == NZ - 1:
                        g = gz_foam if foam_z1 else gz_bare
                        if g > 0:
                            diag += dt * g
                            self._bc_rhs[n] += g * self.T_amb

                    rows.append(n); cols.append(n); vals.append(self._C[n] + diag)

        self._A = csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        self._lu = splu(self._A.tocsc())
        self._dt = dt

    def _gy_conductance(self, n, nb, Ay):
        kA = self._ky[n]; kB = self._ky[nb]
        dA = self._dy[n]; dB = self._dy[nb]
        denom = (dA / 2.0) / kA + (dB / 2.0) / kB
        return Ay / denom

    def step(self, Q, dt, t=None):
        if t is None:
            t = self._t + dt
        Q = np.asarray(Q, dtype=float)
        if Q.size != self.n_cells:
            raise ValueError(f"Q 长度须为 n_cells={self.n_cells}，收到 {Q.size}")
        if self._A is None or self._dt is None or abs(self._dt - dt) > 1e-12:
            self._build(dt)
        src = np.zeros(self.N)
        for n in range(self.N):
            c = self._cell[n]
            if c >= 0:
                src[n] = Q[c] * self._vol[n] / self._V_cell
        rhs = self._C * self.T_field + dt * (src + self._bc_rhs)
        self.T_field = self._lu.solve(rhs)
        self._t = t
        for c in range(self.n_cells):
            mask = self._cell == c
            self.T[c] = float(self.T_field[mask].mean()) if mask.any() else self.T_init
        return self.T.copy()

    def temperature_stats(self):
        stats = {}
        T_max_all = -np.inf
        T_min_all = np.inf
        for c in range(self.n_cells):
            mask = self._cell == c
            tc = self.T_field[mask]
            stats[f"T_avg_cell{c+1} [K]"] = float(tc.mean())
            stats[f"T_max_cell{c+1} [K]"] = float(tc.max())
            T_max_all = max(T_max_all, float(tc.max()))
            T_min_all = min(T_min_all, float(tc.min()))
        stats["T_max [K]"] = T_max_all
        stats["T_min [K]"] = T_min_all
        stats["dT_max [K]"] = T_max_all - T_min_all
        return stats

    def y_profile(self):
        T3d = self.reshape()
        T_y = T3d.mean(axis=(0, 2))
        pos = np.zeros(self.NY)
        y_acc = 0.0
        for j, pj in enumerate(self._per_j):
            pos[j] = y_acc + pj["dy"] / 2.0
            y_acc += pj["dy"]
        return pos, T_y

    def plot_y_profile(self, ax=None, cmap_cell="tab10", show_cell_bounds=True):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        pos, T_y = self.y_profile()
        colors = plt.cm.tab10(np.linspace(0, 1, max(self.n_cells, 10)))
        seg_colors = [colors[pj["cell"] % 10] for j, pj in enumerate(self._per_j)]
        for j in range(self.NY - 1):
            ax.fill_between([pos[j], pos[j + 1]], [T_y[j], T_y[j + 1]],
                            y2=T_y.min() - 5, color=seg_colors[j], alpha=0.35, step="mid")
        ax.plot(pos, T_y, "k-", lw=1.2)
        if show_cell_bounds:
            y_acc = 0.0
            for c in range(self.n_cells):
                y0 = y_acc
                y_acc += self._per_j[c * self.ny]["dy"] * self.ny
                ax.text((y0 + y_acc) / 2.0, T_y.max() + 1.0, f"bat{c+1}",
                        ha="center", fontsize=8)
        ax.set(xlabel="Y Position [m]", ylabel="Temperature [K]",
               title=f"Y-Profile (t={self._t:.0f}s)")
        ax.grid(alpha=0.3)
        return ax.figure, ax

    def plot_xz_slice(self, z_frac=0.5, ax=None, cmap="hot"):
        import matplotlib.pyplot as plt
        T3d = self.reshape()
        k = int(round(z_frac * (self.NZ - 1)))
        k = max(0, min(k, self.NZ - 1))
        slc = T3d[:, :, k]
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(slc.T, origin="lower", aspect="auto",
                       extent=(0, self.Lx * 1000, 0,
                               sum(p["dy"] for p in self._per_j) * 1000),
                       cmap=cmap)
        ax.set(xlabel="X [mm]", ylabel="Y (Stack) [mm]",
               title=f"X-Y Temperature @ Z={k/(self.NZ-1)*self.Lz*1000:.0f}mm")
        plt.colorbar(im, ax=ax, label="Temperature [K]")
        return ax.figure, ax

    def plot_summary(self, save_path=None, dpi=120):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        self.plot_y_profile(ax=axes[0])
        self.plot_xz_slice(ax=axes[1])
        stats = self.temperature_stats()
        fig.suptitle(
            f"3D Stack Temp (t={self._t:.0f}s)  "
            f"T_max={stats['T_max [K]']:.2f}K  "
            f"T_min={stats['T_min [K]']:.2f}K  "
            f"dT={stats['dT_max [K]']:.3f}K",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        if save_path:
            fig.savefig(save_path, dpi=dpi)
        return fig

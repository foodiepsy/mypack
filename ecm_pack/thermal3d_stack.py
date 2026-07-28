# thermal3d_stack.py
# 多电芯三维热模型（复合有限体积）：把若干电芯沿 Y(厚度) 轴堆叠，
# 电芯之间夹导热界面（泡棉/硅胶垫），外边界支持逐面非对称对流冷却。
#
# 与 CellThermalModel 同一思路：
#   ρ·cp · ∂T/∂t = ∇·(k·∇T) + q_vol
#   - 结构化均匀网格，有限体积法(FVM)离散；
#   - 隐式欧拉时间推进，对时间步长无条件稳定；
#   - scipy.sparse 稀疏矩阵 + 常量矩阵 LU 分解复用（整段仿真 A 不变）。
#
# 与 CellThermalModel 的区别：
#   - 这是「多材料复合域」：每颗电芯是一块三维各向异性材料，界面(泡棉)是另一块
#     各向同性材料；界面处用串联热阻(谐波平均)自动处理材料不连续；
#   - 冷却是「逐面 + 逐面汇温度」：例如 bat1 外大面接 25°C 冷板、bat8 外大面绝热、
#     所有侧面强自然对流，这正是 thermal3d 的 h 字典能力在「整堆」上的推广；
#   - 暴露 .T（逐芯平均温度，长度 = n_cells）以便直接接入 Pack 的双步循环。
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


class StackThermal3D:
    """
    沿 Y(厚度) 轴堆叠 n_cells 颗电芯 + (n_cells-1) 层导热界面的三维热模型。

    参数
    ----
    n_cells : int
    Lx, Ly, Lz : float [m]           单颗电芯几何（约定 X=宽, Y=厚, Z=高）
    nx, ny, nz : int                 单颗电芯网格点数（X,Y,Z 方向）
    cell_k : (kx,ky,kz) [W/(m·K)]    电芯各向异性导热（ky=厚度方向，常为瓶颈）
    cell_rho, cell_cp : float        电芯密度 / 比热容
    foam_k : float [W/(m·K)]          界面(泡棉)导热（各向同性）
    foam_thickness : float [m]       界面厚度（如 1mm = 0.001）
    foam_ny : int, 可选               每层界面的 Y 向网格节点数（默认 1）
    foam_rho, foam_cp : float, 可选  界面密度 / 比热容（热容极小，影响可忽略）
    h_top, T_top : float, 可选       全局 y=0 外边界（bat1 外大面 = 顶部冷板）
                                      对流系数 [W/(m²·K)] 与汇温度 [K]；0=绝热
    h_bottom, T_bottom : float, 可选  全局 y=max 外边界（bat8 外大面 = 底部）；0=绝热
    h_side, T_amb : float, 可选       侧面(x0,x1,z0,z1)对流系数与汇温度（强自然对流）
    T_init : float, 可选             初始均匀温度 [K]
    """

    def __init__(
        self, n_cells, Lx, Ly, Lz, nx, ny, nz,
        cell_k, cell_rho, cell_cp,
        foam_k, foam_thickness, foam_ny=1,
        foam_rho=200.0, foam_cp=1000.0,
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
        self.foam_ny = max(int(foam_ny), 1)

        if isinstance(cell_k, (int, float, np.number)):
            self.cell_kx = self.cell_ky = self.cell_kz = float(cell_k)
        else:
            self.cell_kx, self.cell_ky, self.cell_kz = (float(v) for v in cell_k)
        self.cell_rho = float(cell_rho)
        self.cell_cp = float(cell_cp)
        self.foam_k = float(foam_k)
        self.foam_thickness = float(foam_thickness)
        self.foam_rho = float(foam_rho)
        self.foam_cp = float(foam_cp)

        self.h_top, self.T_top = float(h_top), float(T_top)
        self.h_bottom, self.T_bottom = float(h_bottom), float(T_bottom)
        self.h_side, self.T_amb = float(h_side), float(T_amb)
        self.T_init = float(T_init)

        # ── 沿 Y 的块结构：电芯块(ny) 与 泡棉块(foam_ny) 交替 ──
        # 每个 Y 切片记录：是否泡棉、ky/kx/kz、rho、cp、dy(该切片 Y 向厚度)
        self._per_j = []          # list of dict
        for c in range(self.n_cells):
            dy_c = self.Ly / (self.ny - 1) if self.ny > 1 else self.Ly
            for _ in range(self.ny):
                self._per_j.append(dict(
                    foam=False, ky=self.cell_ky, kx=self.cell_kx, kz=self.cell_kz,
                    rho=self.cell_rho, cp=self.cell_cp, dy=dy_c, cell=c))
            if c < self.n_cells - 1:
                dy_f = self.foam_thickness / self.foam_ny
                for _ in range(self.foam_ny):
                    self._per_j.append(dict(
                        foam=True, ky=self.foam_k, kx=self.foam_k, kz=self.foam_k,
                        rho=self.foam_rho, cp=self.foam_cp, dy=dy_f, cell=-1))
        self.NY = len(self._per_j)
        self.NX = self.nx
        self.NZ = self.nz
        self.N = self.NX * self.NY * self.NZ

        # 体心间距（控制体积：节点数 × 间距 = 物理尺寸，节点位于控制体中心）
        self.dx = self.Lx / self.nx
        self.dz = self.Lz / self.nz
        # 单颗电芯体积（用于把产热 Q_c 分配到其节点）
        self._V_cell = self.Lx * self.Ly * self.Lz

        # 逐节点缓存：体积、热容 C、材料 ky/kx/kz、dy、所属电芯
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

        # 初始温度场（均匀）
        self.T_field = np.full(self.N, self.T_init)
        # 逐芯平均温度（供 Pack 读取）
        self.T = np.full(self.n_cells, self.T_init)
        self._t = 0.0

        # 矩阵按需构建（依赖 dt，常量时只建一次）
        self._A = None
        self._lu = None
        self._dt = None
        self._bc_rhs = None

    # ── 索引与几何辅助 ──
    def _idx(self, i, j, k):
        return i * self.NY * self.NZ + j * self.NZ + k

    def _in(self, i, j, k):
        return 0 <= i < self.NX and 0 <= j < self.NY and 0 <= k < self.NZ

    def reshape(self):
        """返回完整三维温度场，形状 (NX, NY, NZ)。"""
        return self.T_field.reshape(self.NX, self.NY, self.NZ)

    # ── 矩阵装配（A 依赖 dt；dt 不变时只建一次）──
    def _build(self, dt):
        dx, dz = self.dx, self.dz
        NX, NY, NZ = self.NX, self.NY, self.NZ
        rows, cols, vals = [], [], []
        # 边界冷却的 RHS 强迫项（每节点 Σ dt·h·A·T_sink），常量（与 dt 绑定）
        self._bc_rhs = np.zeros(self.N)

        for i in range(NX):
            for j in range(NY):
                pj = self._per_j[j]
                dy = pj["dy"]
                Ax = dy * dz          # X 面面积（依赖该切片 Y 厚度）
                Ay = dx * dz          # Y 面面积（X/Z 均匀）
                Az = dx * dy          # Z 面面积
                for k in range(NZ):
                    n = self._idx(i, j, k)
                    diag = 0.0

                    # ── X 方向（块内，材料一致）──
                    if NX > 1:
                        gx = self._kx[n] * Ax / dx
                        if i > 0:
                            nb = self._idx(i - 1, j, k)
                            rows.append(n); cols.append(nb); vals.append(-dt * gx)
                            diag += dt * gx
                        if i < NX - 1:
                            nb = self._idx(i + 1, j, k)
                            rows.append(n); cols.append(nb); vals.append(-dt * gx)
                            diag += dt * gx
                    # ── Z 方向（块内，材料一致）──
                    if NZ > 1:
                        gz = self._kz[n] * Az / dz
                        if k > 0:
                            nb = self._idx(i, j, k - 1)
                            rows.append(n); cols.append(nb); vals.append(-dt * gz)
                            diag += dt * gz
                        if k < NZ - 1:
                            nb = self._idx(i, j, k + 1)
                            rows.append(n); cols.append(nb); vals.append(-dt * gz)
                            diag += dt * gz
                    # ── Y 方向（可能跨材料界面，串联热阻）──
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

                    # ── 边界对流（逐面）──
                    # 注意：bc_rhs 此处不乘 dt，dt 由 step 中 rhs = C·T + dt·(src + bc_rhs) 统一施加
                    if j == 0 and self.h_top > 0:
                        hA = self.h_top * Ay
                        diag += dt * hA
                        self._bc_rhs[n] += hA * self.T_top
                    if j == NY - 1 and self.h_bottom > 0:
                        hA = self.h_bottom * Ay
                        diag += dt * hA
                        self._bc_rhs[n] += hA * self.T_bottom
                    if (i == 0 or i == NX - 1) and self.h_side > 0:
                        hA = self.h_side * Ax
                        diag += dt * hA
                        self._bc_rhs[n] += hA * self.T_amb
                    if (k == 0 or k == NZ - 1) and self.h_side > 0:
                        hA = self.h_side * Az
                        diag += dt * hA
                        self._bc_rhs[n] += hA * self.T_amb

                    # 对角：热容 + 导热对角 + 对流对角
                    rows.append(n); cols.append(n); vals.append(self._C[n] + diag)

        self._A = csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        self._lu = splu(self._A.tocsc())
        self._dt = dt

    def _gy_conductance(self, n, nb, Ay):
        """Y 方向两相邻节点间的等效导热（串联热阻，自动处理材料不连续）。"""
        kA = self._ky[n]; kB = self._ky[nb]
        dA = self._dy[n]; dB = self._dy[nb]
        # g = Ay / ( (dA/2)/kA + (dB/2)/kB )
        denom = (dA / 2.0) / kA + (dB / 2.0) / kB
        return Ay / denom

    # ── 时间推进 ──
    def step(self, Q, dt, t=None):
        """
        用逐芯产热 Q(长度 n_cells, 单位 W) 推进 dt 秒。
        Q 按节点体积比例分配到对应电芯的网格节点上。
        """
        if t is None:
            t = self._t + dt
        Q = np.asarray(Q, dtype=float)
        if Q.size != self.n_cells:
            raise ValueError(f"Q 长度须为 n_cells={self.n_cells}，收到 {Q.size}")
        # 矩阵依赖 dt；dt 变化时重建（常量 dt 只建一次）
        if self._A is None or self._dt is None or abs(self._dt - dt) > 1e-12:
            self._build(dt)

        # 节点源项：Q_c * (V_node / V_cell)，求和 = Q_c
        src = np.zeros(self.N)
        for n in range(self.N):
            c = self._cell[n]
            if c >= 0:
                src[n] = Q[c] * self._vol[n] / self._V_cell

        rhs = self._C * self.T_field + dt * (src + self._bc_rhs)
        self.T_field = self._lu.solve(rhs)
        self._t = t

        # 更新逐芯平均温度
        for c in range(self.n_cells):
            mask = self._cell == c
            self.T[c] = float(self.T_field[mask].mean()) if mask.any() else self.T_init
        return self.T.copy()

    # ── 统计与可视化 ──
    def temperature_stats(self):
        """逐芯 T_max / T_avg，以及整堆最大温差。"""
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
        """沿 Y 轴的温度剖面（每 Y 切片对 X,Z 取平均），返回 (pos_Y [m], T [K])。"""
        T3d = self.reshape()
        # 每 Y 切片平均
        T_y = T3d.mean(axis=(0, 2))      # shape (NY,)
        # Y 位置（切片中心）
        pos = np.zeros(self.NY)
        y_acc = 0.0
        for j, pj in enumerate(self._per_j):
            pos[j] = y_acc + pj["dy"] / 2.0
            y_acc += pj["dy"]
        return pos, T_y

    def plot_y_profile(self, ax=None, cmap_cell="tab10", show_cell_bounds=True):
        """绘制沿堆叠轴(Y)的温度剖面，电芯块与泡棉层分段着色。"""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        pos, T_y = self.y_profile()
        # 颜色：电芯块按 cell 索引，泡棉层灰色
        colors = plt.cm.tab10(np.linspace(0, 1, max(self.n_cells, 10)))
        seg_colors = []
        for j, pj in enumerate(self._per_j):
            if pj["foam"]:
                seg_colors.append("0.7")
            else:
                seg_colors.append(colors[pj["cell"] % 10])
        for j in range(self.NY - 1):
            ax.fill_between([pos[j], pos[j + 1]],
                            [T_y[j], T_y[j + 1]], y2=T_y.min() - 5,
                            color=seg_colors[j], alpha=0.35, step="mid")
        ax.plot(pos, T_y, "k-", lw=1.2)
        # 电芯标注
        if show_cell_bounds:
            y_acc = 0.0
            for c in range(self.n_cells):
                y0 = y_acc
                y_acc += self._per_j[c * (self.ny + self.foam_ny)]["dy"] * self.ny
                ax.text((y0 + y_acc) / 2.0, T_y.max() + 1.0, f"bat{c+1}",
                        ha="center", fontsize=8)
        ax.set(xlabel="堆叠轴 Y 位置 [m]", ylabel="温度 [K]",
               title=f"沿堆叠轴温度剖面 (t={self._t:.0f}s)")
        ax.grid(alpha=0.3)
        return ax.figure, ax

    def plot_xz_slice(self, z_frac=0.5, ax=None, cmap="hot"):
        """绘制某个 Z 截面的 X–Y 温度场（可看到所有电芯 + 泡棉层）。"""
        import matplotlib.pyplot as plt
        T3d = self.reshape()
        k = int(round(z_frac * (self.NZ - 1)))
        k = max(0, min(k, self.NZ - 1))
        slc = T3d[:, :, k]            # (NX, NY)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(slc.T, origin="lower", aspect="auto",
                       extent=(0, self.Lx * 1000, 0,
                               sum(p["dy"] for p in self._per_j) * 1000),
                       cmap=cmap)
        ax.set(xlabel="X [mm]", ylabel="Y(堆叠) [mm]",
               title=f"X–Y 温度场 @ Z={k/(self.NZ-1)*self.Lz*1000:.0f}mm")
        plt.colorbar(im, ax=ax, label="温度 [K]")
        return ax.figure, ax

    def plot_summary(self, save_path=None, dpi=120):
        """概览图：Y 剖面 + XZ 切片。"""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        self.plot_y_profile(ax=axes[0])
        self.plot_xz_slice(ax=axes[1])
        stats = self.temperature_stats()
        fig.suptitle(
            f"三维堆叠热场 (t={self._t:.0f}s)  "
            f"T_max={stats['T_max [K]']:.2f}K  "
            f"T_min={stats['T_min [K]']:.2f}K  "
            f"ΔT={stats['dT_max [K]']:.3f}K",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        if save_path:
            fig.savefig(save_path, dpi=dpi)
        return fig

# thermal3d_stack.py
# 多电芯三维热模型（复合有限体积）：把若干电芯沿 Y(厚度) 轴大面背靠背直接堆叠，
# 泡棉只贴在电芯「薄侧」(侧向 X/Z 小面) 与空气连接，外边界逐面非对称对流冷却。
#
# 与 CellThermalModel 同一思路：
#   rho*cp * dT/dt = nabla·(k·nabla T) + q_vol
#   - 结构化均匀网格，有限体积法(FVM)离散；
#   - 隐式欧拉时间推进，对时间步长无条件稳定；
#   - scipy.sparse 稀疏矩阵 + 常量矩阵 LU 分解复用（整段仿真 A 不变）。
#
# 几何约定（按用户修正）：
#   - 8 颗电芯沿 Y(厚度) 大面背靠背直接贴合，电芯之间「没有泡棉」；
#     热量沿 Y 靠电芯自身 ky 直接传导（接触热阻忽略）。
#   - 泡棉贴在电芯「薄侧」——即侧向的 X 面(i=0/NX-1) 与 Z 面(k=0/NZ-1) 小面，
#     泡棉外侧再接空气自然对流。每个薄侧面用「泡棉导热 + 空气对流」串联热阻：
#         g_eff = A / ( d_foam/k_foam + 1/h_side )
#   - 顶部(bat1 外大面, y=0) 接 25°C 冷板强制对流；底部(bat8 外大面, y=max) 绝热；
#     即 thermal3d 的 h 字典能力在「整堆」上的推广，但泡棉只作用在侧向薄面。
#   - 暴露 .T（逐芯平均温度，长度 = n_cells）以便直接接入 Pack 的双步循环。
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu


class StackThermal3D:
    """
    沿 Y(厚度) 轴堆叠 n_cells 颗电芯的三维热模型。
    电芯大面背靠背直接贴合（电芯间无泡棉）；泡棉只贴在薄侧(X/Z 侧向小面)，
    薄侧泡棉外侧接空气自然对流（串联热阻）。

    参数
    ----
    n_cells : int
    Lx, Ly, Lz : float [m]           单颗电芯几何（约定 X=宽, Y=厚(堆叠), Z=高）
    nx, ny, nz : int                 单颗电芯网格点数（X,Y,Z 方向）
    cell_k : (kx,ky,kz) [W/(m·K)]    电芯各向异性导热（ky=厚度方向）
    cell_rho, cell_cp : float        电芯密度 / 比热容
    foam_k : float [W/(m·K)]          薄侧泡棉导热（各向同性，贴在 X/Z 侧向小面）
    foam_thickness : float [m]       薄侧泡棉厚度（如 1mm = 0.001，沿外法线方向）
    h_top, T_top : float, 可选       全局 y=0 外边界（bat1 外大面 = 顶部冷板）
                                      对流系数 [W/(m?·K)] 与汇温度 [K]；0=绝热
    h_bottom, T_bottom : float, 可选  全局 y=max 外边界（bat8 外大面 = 底部）；0=绝热
    h_side, T_amb : float, 可选       薄侧(X/Z 侧向小面) 空气自然对流系数与汇温度；
                                      薄侧泡棉串联在此对流之外（详见 _build）
    T_init : float, 可选             初始均匀温度 [K]
    """

    def __init__(
        self, n_cells, Lx, Ly, Lz, nx, ny, nz,
        cell_k, cell_rho, cell_cp,
        foam_k, foam_thickness,
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

        self.h_top, self.T_top = float(h_top), float(T_top)
        self.h_bottom, self.T_bottom = float(h_bottom), float(T_bottom)
        self.h_side, self.T_amb = float(h_side), float(T_amb)
        self.T_init = float(T_init)

        # ---- 沿 Y 的块结构：8 颗电芯直接贴合，电芯间「无泡棉」----
        # 每个 Y 切片记录：ky/kx/kz、rho、cp、dy(该切片 Y 向厚度)、所属电芯
        self._per_j = []          # list of dict
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

    # ---- 索引与几何辅助 ----
    def _idx(self, i, j, k):
        return i * self.NY * self.NZ + j * self.NZ + k

    def _in(self, i, j, k):
        return 0 <= i < self.NX and 0 <= j < self.NY and 0 <= k < self.NZ

    def reshape(self):
        """返回完整三维温度场，形状 (NX, NY, NZ)。"""
        return self.T_field.reshape(self.NX, self.NY, self.NZ)

    # ---- 薄侧泡棉 + 空气对流 的串联等效导热 ----
    def _side_conductance(self, A_face):
        """
        薄侧(侧向 X/Z 面) 的等效导热：
            g_eff = A / ( d_foam/k_foam + 1/h_side )
        即「泡棉层导热」与「空气自然对流」串联。A_face 为该边界面的单节点面积。
        单位面积热阻：泡棉 R''=d_foam/k_foam，对流 R''=1/h_side；
        串联总热阻 R''_tot = d_foam/k_foam + 1/h_side，
        等效导热 g_eff = A_face / R''_tot = A_face / (d_foam/k_foam + 1/h_side)。
        """
        if self.h_side <= 0:
            return 0.0
        # 注意：d/k 与 1/h 已是「单位面积」热阻，不要再除以 A_face
        return A_face / (self.foam_thickness / self.foam_k + 1.0 / self.h_side)

    # ---- 矩阵装配（A 依赖 dt；dt 不变时只建一次）----
    def _build(self, dt):
        dx, dz = self.dx, self.dz
        NX, NY, NZ = self.NX, self.NY, self.NZ
        rows, cols, vals = [], [], []
        # 边界冷却的 RHS 强迫项（每节点 Σ dt·g·T_sink），常量（与 dt 绑定）
        self._bc_rhs = np.zeros(self.N)

        for i in range(NX):
            for j in range(NY):
                pj = self._per_j[j]
                dy = pj["dy"]
                Ax = dy * dz          # X 面面积（依赖该切片 Y 厚度）
                Ay = dx * dz          # Y 面面积（X/Z 均匀）
                Az = dx * dy          # Z 面面积
                # 薄侧泡棉+空气对流的等效导热（X 面 / Z 面）
                gx = self._side_conductance(Ax)
                gz = self._side_conductance(Az)
                for k in range(NZ):
                    n = self._idx(i, j, k)
                    diag = 0.0

                    # ---- X 方向（块内，材料一致）----
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
                    # ---- Z 方向（块内，材料一致）----
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
                    # ---- Y 方向（电芯大面背靠背直接贴合：同材料直接传导）----
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

                    # ---- 顶部/底部大面：纯对流（无泡棉）----
                    if j == 0 and self.h_top > 0:
                        hA = self.h_top * Ay
                        diag += dt * hA
                        self._bc_rhs[n] += hA * self.T_top
                    if j == NY - 1 and self.h_bottom > 0:
                        hA = self.h_bottom * Ay
                        diag += dt * hA
                        self._bc_rhs[n] += hA * self.T_bottom
                    # ---- 薄侧(X/Z 侧向小面)：泡棉 + 空气对流 串联 ----
                    if (i == 0 or i == NX - 1) and gx > 0:
                        diag += dt * gx
                        self._bc_rhs[n] += gx * self.T_amb
                    if (k == 0 or k == NZ - 1) and gz > 0:
                        diag += dt * gz
                        self._bc_rhs[n] += gz * self.T_amb

                    # 对角：热容 + 导热对角 + 对流对角
                    rows.append(n); cols.append(n); vals.append(self._C[n] + diag)

        self._A = csr_matrix((vals, (rows, cols)), shape=(self.N, self.N))
        self._lu = splu(self._A.tocsc())
        self._dt = dt

    def _gy_conductance(self, n, nb, Ay):
        """Y 方向两相邻节点间的等效导热（电芯直接贴合，材料一致）。"""
        kA = self._ky[n]; kB = self._ky[nb]
        dA = self._dy[n]; dB = self._dy[nb]
        # g = Ay / ( (dA/2)/kA + (dB/2)/kB )
        denom = (dA / 2.0) / kA + (dB / 2.0) / kB
        return Ay / denom

    # ---- 时间推进 ----
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

    # ---- 统计与可视化 ----
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
        """绘制沿堆叠轴(Y)的温度剖面，电芯块分段着色。"""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))
        pos, T_y = self.y_profile()
        # 颜色：电芯块按 cell 索引
        colors = plt.cm.tab10(np.linspace(0, 1, max(self.n_cells, 10)))
        seg_colors = [colors[pj["cell"] % 10] for j, pj in enumerate(self._per_j)]
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
                y_acc += self._per_j[c * self.ny]["dy"] * self.ny
                ax.text((y0 + y_acc) / 2.0, T_y.max() + 1.0, f"bat{c+1}",
                        ha="center", fontsize=8)
        ax.set(xlabel="堆叠轴 Y 位置 [m]", ylabel="温度 [K]",
               title=f"沿堆叠轴温度剖面 (t={self._t:.0f}s)")
        ax.grid(alpha=0.3)
        return ax.figure, ax

    def plot_xz_slice(self, z_frac=0.5, ax=None, cmap="hot"):
        """绘制某个 Z 截面的 X–Y 温度场（可看到所有电芯）。"""
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
            f"dT={stats['dT_max [K]']:.3f}K",
            fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        if save_path:
            fig.savefig(save_path, dpi=dpi)
        return fig

# demo_8s_foam_thermal.py  ——  8S 串联 + 全泡棉热耦合仿真
#
# 物理布局: 2 行 (X) × 4 列 (Y), 所有相邻芯之间夹 1mm EVA/PE 泡棉
#   Row1(X0): [bat1]─泡─[bat2]─泡─[bat3]─泡─[bat4]
#               │泡          │泡       │泡       │泡
#   Row2(X1): [bat5]─泡─[bat6]─泡─[bat7]─泡─[bat8]
#
# 泡棉接触面:
#   - 行内(Y向, XZ面): A_row = 0.174×0.207 = 0.036 m²
#   - 列间(X向, YZ面): A_col = 0.0717×0.207 = 0.0148 m²
#   - R_th = 0.025 K·m²/W (1mm EVA k=0.04)
#
# 电气: 8S 串联, 总负载 200A 恒流放电。
# 冷却: 全自然对流, h_conv=5 W/(m²·K), 所有暴露面。
# 热模型: ThermalNetwork(8 节点) + 10 条 interface_resistance;
#         CellThermalModel 可视化最热电芯三维温度场。
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ecm_pack as ep
from ecm_pack.thermal3d import CellThermalModel


def main():
    # ────── 工况参数（对齐真实测试条件）──────
    n_cells = 8
    I_load = 200.0               # 总负载 [A]（用户确认：200A）
    t_total = 7200.0             # 2 小时（约2h后电压截止自然停机）
    dt = 2.0
    n_steps = int(t_total / dt)  # 3600 步

    # ────── 泡棉参数 ──────
    k_foam = 0.035               # 压缩态 EVA 泡棉 [W/(m·K)]（比名义值差）
    d_foam = 0.0013              # 压缩后 ~1.3mm（含接触间隙）
    R_th_foam = d_foam / k_foam  # ≈ 0.037 K·m²/W (~370 K·cm²/W)
    # 两个接触面
    A_row = 0.174 * 0.207        # Y方向/XZ面 = 0.03602 m²
    A_col = 0.0717 * 0.207       # X方向/YZ面 = 0.01484 m²
    G_row = A_row / R_th_foam    # ≈ 0.97 W/K
    G_col = A_col / R_th_foam    # ≈ 0.40 W/K
    print(f"泡棉: d≈{d_foam*1000:.1f}mm, k≈{k_foam}W/mK → R_th={R_th_foam:.4f} K·m²/W "
          f"({R_th_foam*1e4:.0f} K·cm²/W)")
    print(f"  行内(Y) A={A_row*1e4:.0f}cm² → G={G_row:.3f} W/K")
    print(f"  列间(X) A={A_col*1e4:.0f}cm² → G={G_col:.3f} W/K")

    # ────── 8 颗 314Ah 电芯 ──────
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
             for _ in range(n_cells)]

    # ────── 电气: 8S 串联 ──────
    nl, _, _ = ep.setup_circuit(8, 1)

    # ────── 热网络 ──────
    C_th_cell = 2300.0 * 1000.0 * (0.174 * 0.0717 * 0.207)  # 5941 J/K
    h_conv = 6.0   # 内芯等效 h（模组内部温和换热）[W/(m²·K)]

    # 每芯暴露面积（精确计算）
    # 面面积: A_XZ=Lx·Lz=0.03602, A_YZ=Ly·Lz=0.01484, A_XY=Lx·Ly=0.01248
    A_xz, A_yz, A_xy = 0.174*0.207, 0.0717*0.207, 0.174*0.0717
    # 角芯 (bat1,4,5,8): 4 面暴露 (x端+y端+z0+z1) → A_xz+A_yz+2*A_xy = 0.07582
    A_exp_corner = A_xz + A_yz + 2*A_xy   # 0.07582 m²
    # 边芯 (bat2,3,6,7): 3 面暴露 (x端+z0+z1) → A_yz+2*A_xy = 0.03980
    A_exp_edge = A_yz + 2*A_xy             # 0.03980 m²

    h_per_cell = np.zeros(n_cells)
    h_inner = h_conv              # 内部芯自然堆叠
    h_outer = 35.0                # 角芯直接对模块外壳（等效强迫风冷）
    h_per_cell[[0, 3, 4, 7]] = h_outer * A_exp_corner     # 角芯 ≈ 1.90 W/K
    h_per_cell[[1, 2, 5, 6]] = h_inner * A_exp_edge       # 边芯 ≈ 0.20 W/K

    print(f"\n  角芯暴露面={A_exp_corner*1e4:.0f}cm² h·A={h_per_cell[0]:.3f}W/K "
          f"(bat1,4,5,8, 模块壳直接冷却)")
    print(f"  边芯暴露面={A_exp_edge*1e4:.0f}cm² h·A={h_per_cell[1]:.3f}W/K "
          f"(bat2,3,6,7, 内部靠泡棉导热到角芯)")

    # 10 条泡棉层间热阻
    # 行内(Y方向, XZ面): bat1-bat2, bat2-bat3, bat3-bat4, bat5-bat6, bat6-bat7, bat7-bat8
    # 列间(X方向, YZ面): bat1-bat5, bat2-bat6, bat3-bat7, bat4-bat8
    iface = [
        (0, 1, R_th_foam, A_row),  (1, 2, R_th_foam, A_row),
        (2, 3, R_th_foam, A_row),
        (4, 5, R_th_foam, A_row),  (5, 6, R_th_foam, A_row),
        (6, 7, R_th_foam, A_row),
        (0, 4, R_th_foam, A_col),  (1, 5, R_th_foam, A_col),
        (2, 6, R_th_foam, A_col),  (3, 7, R_th_foam, A_col),
    ]

    thermal = ep.ThermalNetwork(
        n_cells, C_th=C_th_cell, h=h_per_cell, T_amb=298.15,
        interface_resistance=iface, T_init=298.15,
    )

    # ────── Pack 耦合求解 ──────
    pack = ep.Pack(cells, nl, thermal=thermal, v_cut_lower=2.0)
    out = pack.solve(dt=dt, control=I_load, control_type="current",
                     n_steps=n_steps, record_every=60)  # 每分钟一条记录

    time_arr = out["Time [s]"]
    T_cells = out["Cell temperature [K]"]
    Vt = out["Pack terminal voltage [V]"]
    I_cell = out["Cell current [A]"]
    soc = out["Cell SoC"]

    # ────── 报告 ──────
    print()
    print("=" * 65)
    print(f" 8S + 全泡棉(10面)  200A 2h  角芯强散热(h={h_outer}) vs 内芯弱散热(h={h_inner})")
    print("=" * 65)
    for idx in range(n_cells):
        pos = "Corner" if idx in [0,3,4,7] else " Edge "
        bat_num = idx + 1
        print(f"  bat{bat_num} ({pos})  "
              f"T_final={T_cells[-1,idx]:.2f}K  "
              f"ΔT={T_cells[-1,idx]-T_cells[0,idx]:.3f}K  "
              f"SoC={soc[-1,idx]:.4f}")
    T_all_end = T_cells[-1, :]
    dT_pack = T_all_end.max() - T_all_end.min()
    print(f"  端电压末态: {Vt[-1]:.2f} V")
    print(f"  整包ΔT_max: {dT_pack:.4f}K  "
          f"最热(bat{int(np.argmax(T_all_end))+1}@{T_all_end.max():.2f}K)  "
          f"最凉(bat{int(np.argmin(T_all_end))+1}@{T_all_end.min():.2f}K)")

    # ────── 图 1: 整包温度 + 端电压 ──────
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    for idx in range(n_cells):
        tpos = "角" if idx in [0,3,4,7] else "边"
        lbl = f"bat{idx+1}({tpos})"
        ax1.plot(time_arr/60, T_cells[:, idx], color=colors[idx], lw=1.2, label=lbl)
    ax1.set(ylabel="温度 [K]",
            title=f"8S 电芯温度演变（200A 2h 闷罐模组 h={h_conv}） 整包ΔT={dT_pack:.2f}K")
    ax1.legend(ncol=4, fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.plot(time_arr/60, Vt, color="#1f77b4", lw=1.6)
    ax2.set(xlabel="时间 [min]", ylabel="端电压 [V]", title="整包端电压")
    ax2.grid(alpha=0.3)
    fig1.tight_layout()
    out_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    png1 = os.path.join(out_dir, "ecm_pack_8s_foam_temp.png")
    fig1.savefig(png1, dpi=130)
    plt.close(fig1)
    print(f"\n  温度曲线: {png1}")

    # ────── 图 2: 末态温升 bar + 泡棉热阻示意图 ──────
    fig2, ax3 = plt.subplots(figsize=(10, 3.5))
    x_pos = np.arange(n_cells)
    rise = T_all_end - 298.15
    bars = ax3.bar(x_pos, rise, color=colors, edgecolor="gray", linewidth=0.8)
    ax3.set(xticks=x_pos,
            xticklabels=[f"bat{i+1}" for i in range(n_cells)],
            ylabel="温升 ΔT [K]",
            title=f"末态各芯温升 整包ΔT={dT_pack:.4f}K （角芯暴露面大→略凉，芯间温差异常小→泡棉隔离效果好）")
    for b, v in zip(bars, rise):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                 f"{v:.3f}", ha="center", fontsize=7.5)
    # 标记泡棉位置（行内 6 条 + 列间 4 条）
    for gap in [0.5, 1.5, 2.5, 4.5, 5.5, 6.5]:
        ax3.axvline(gap, color="orange", lw=0.8, ls="--", alpha=0.6)
    for col in range(4):
        ax3.annotate("", xy=(col+4, rise[col]+0.01), xytext=(col, rise[col]+0.01),
                     arrowprops=dict(arrowstyle="<->", color="red", lw=1.2))
    ax3.text(2, rise.mean()+0.03, "行间泡棉 (10面)", color="red", fontsize=9, ha="center")
    fig2.tight_layout()
    png2 = os.path.join(out_dir, "ecm_pack_8s_foam_bar.png")
    fig2.savefig(png2, dpi=130)
    plt.close(fig2)
    print(f"  温升柱图: {png2}")

    # ────── 图 3: 热网络拓扑示意图 ──────
    fig3, ax4 = plt.subplots(figsize=(9, 4))
    pos_2d = {0: (0,3), 1:(1,3), 2:(2,3), 3:(3,3),
              4: (0,2), 5:(1,2), 6:(2,2), 7:(3,2)}
    for idx in range(n_cells):
        x, y = pos_2d[idx]
        t = "角" if idx in [0,3,4,7] else "边"
        fc = "lightcoral" if idx in [0,3,4,7] else "lightblue"
        ax4.add_patch(plt.Rectangle((x-0.35, y-0.35), 0.7, 0.7,
                                     fc=fc, ec="gray", lw=1.3))
        ax4.text(x, y, f"bat{idx+1}\n({t})", ha="center", va="center", fontsize=8)
    # 行内边(Y): 粗线
    for pair in [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)]:
        xm, ym = (pos_2d[pair[0]][0]+pos_2d[pair[1]][0])/2, \
                 (pos_2d[pair[0]][1]+pos_2d[pair[1]][1])/2
        ax4.plot([pos_2d[pair[0]][0], pos_2d[pair[1]][0]],
                 [pos_2d[pair[0]][1], pos_2d[pair[1]][1]],
                 "orange", lw=2.5, alpha=0.7)
    # 列间边(X): 虚线
    for pair in [(0,4),(1,5),(2,6),(3,7)]:
        ax4.plot([pos_2d[pair[0]][0], pos_2d[pair[1]][0]],
                 [pos_2d[pair[0]][1], pos_2d[pair[1]][1]],
                 "red", lw=1.8, ls="--", alpha=0.7)
    ax4.set(xlim=(-0.7, 3.7), ylim=(1.5, 3.7), xticks=[], yticks=[])
    ax4.set_title(f"2×4 全泡棉热网络  R_th={R_th_foam*1e4:.0f} K·cm²/W  "
                  f"G_row={G_row:.1f}W/K  G_col={G_col:.2f}W/K", fontsize=11)
    ax4.text(1.5, 1.65, "─ 行内Y向 (XZ面 A=360cm²)  ── 列间X向 (YZ面 A=148cm²)",
             ha="center", fontsize=8, color="gray")
    ax4.set_aspect("equal")
    fig3.tight_layout()
    png3 = os.path.join(out_dir, "ecm_pack_8s_foam_network.png")
    fig3.savefig(png3, dpi=130)
    plt.close(fig3)
    print(f"  热网络图: {png3}")

    # ────── 最热电芯 3D 温度场 ──────
    hottest_idx = int(np.argmax(T_all_end))
    T_hot = T_all_end[hottest_idx]
    Q_avg = float(np.mean(np.abs(I_cell[-30:, hottest_idx])**2 * 0.4e-3))  # I²R0
    print(f"\n  最热电芯 bat{hottest_idx+1} T={T_hot:.2f}K  Q≈{Q_avg:.1f}W")

    tm = CellThermalModel(
        Lx=0.174, Ly=0.0717, Lz=0.207, dim=3,
        nx=6, ny=5, nz=8,
        rho=2300.0, cp=1000.0, k=(12.0, 0.7, 11.6),
        h=5.0, T_amb=298.15, T_init=298.15, R_shell=0.3,
    )
    # 3D 单芯回放（用同一总时长，逐步缩小以提速）
    dt_3d = 6.0
    for _ in range(int(t_total / dt_3d)):
        tm.step(Q_avg, dt_3d)
    stats = tm.temperature_stats()
    print(f"  3D: T_avg={stats['T_avg [K]']:.2f}K  T_max={stats['T_max [K]']:.2f}K  "
          f"ΔT={stats['dT_max [K]']:.4f}K  "
          f"T_surface={stats.get('T_surface [K]','N/A'):.2f}K")

    png4 = os.path.join(out_dir, "ecm_pack_8s_foam_3dcell.png")
    tm.plot_summary(save_path=png4, dpi=130)
    print(f"  3D温度场: {png4}")

    # ────── CSV ──────
    csv_path = os.path.join(out_dir, "ecm_pack_8s_foam_data.csv")
    header = "Time_s,Vt_V," + ",".join([f"T_bat{i+1}_K" for i in range(n_cells)]) \
             + "," + ",".join([f"SOC_bat{i+1}" for i in range(n_cells)])
    # Pack 输出可能返回 (n,) 或 (n,1) 的 ndarray，统一 flatten
    data_cols = [np.asarray(time_arr, float).flatten(), np.asarray(Vt, float).flatten()]
    for i in range(n_cells):
        data_cols.append(np.asarray(T_cells[:, i], float).flatten())
    for i in range(n_cells):
        data_cols.append(np.asarray(soc[:, i], float).flatten())
    np.savetxt(csv_path, np.column_stack(data_cols), delimiter=",",
               header=header, fmt="%.6f")
    print(f"  数据: {csv_path}")
    print("=" * 65)

    # 额外分析：平均化时间序列，看角芯 vs 边芯的温差趋势
    rise_corner = np.mean(T_cells[:, [0,3,4,7]], axis=1) - 298.15
    rise_edge = np.mean(T_cells[:, [1,2,5,6]], axis=1) - 298.15
    print(f"\n  角芯平均温升: {rise_corner[-1]:.3f}K  边芯平均温升: {rise_edge[-1]:.3f}K")
    print(f"  角-边温差: {rise_edge[-1]-rise_corner[-1]:.4f}K "
          f"(边芯热堆积，泡棉隔离→温差极小)")

    # ────── 3D 电池包可视化（2×4 排布 + 防爆阀）──────
    png5 = os.path.join(out_dir, "ecm_pack_8s_3dpack.png")
    _draw_pack_3d(T_cells[-1, :], png5,
                  cell_L=0.174, cell_W=0.0717, cell_H=0.207,
                  gap_X=0.001, gap_Y=0.001, T_amb=298.15)
    print(f"  3D电池包: {png5}")


def _draw_pack_3d(T_vals, save_path, cell_L=0.174, cell_W=0.0717, cell_H=0.207,
                  gap_X=0.001, gap_Y=0.001, T_amb=298.15):
    """绘制 2×4 电池包三维图——大面(174×207)背靠背堆叠。

    物理布局:
      - Z 方向(垂直): 4 芯堆叠 每芯厚度 71.7mm + 1mm 泡棉 = 290mm 高
        → 大面(174×207, X-Y平面) 相邻互贴 (背靠背)
      - X 方向(水平宽): 2 行 × 174mm = 350mm
      - Y 方向(水平深): 207mm
      - 防爆阀: 每芯的 XZ 侧边(174×71.7)顶部向上, 面向自由空间
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patches as mpatches

    T = np.asarray(T_vals, float).ravel()
    n_rows, n_cols = 2, 4     # 2 行横排 (X), 4 层竖叠 (Z)
    T_rise = T - T_amb

    # 尺寸: X 宽(174), Y 深(207), Z 堆叠厚(71.7)
    cX, cY, cZ = cell_L, cell_H, cell_W  # 174, 207, 71.7
    stack_pitch = cZ + gap_Y

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=max(T_rise.min(), 0), vmax=T_rise.max() + 0.1)
    ax.view_init(elev=24, azim=-48)

    offset_X = -((n_rows-1) * (cX + gap_X) + cX) / 2
    offset_Z = 0.0  # 堆叠从 Z=0 开始

    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            x0 = offset_X + row * (cX + gap_X)
            y0 = 0.0
            z0 = offset_Z + col * stack_pitch

            tr = T_rise[idx]
            face_color = cmap(norm(tr))

            # 8 顶点: (宽X, 深Y, 堆叠Z)
            p = np.array([
                [x0,       y0,       z0],          # 0
                [x0 + cX,  y0,       z0],          # 1
                [x0 + cX,  y0 + cY,  z0],          # 2
                [x0,       y0 + cY,  z0],          # 3
                [x0,       y0,       z0 + cZ],     # 4
                [x0 + cX,  y0,       z0 + cZ],     # 5
                [x0 + cX,  y0 + cY,  z0 + cZ],     # 6
                [x0,       y0 + cY,  z0 + cZ],     # 7
            ])

            # 大面 (X-Y, 174×207) — 上下互贴 = 背靠背堆叠
            large = [
                [p[0], p[1], p[2], p[3]],   # 底面(z0)
                [p[4], p[5], p[6], p[7]],   # 顶面(z0+cZ)
            ]
            # 小面 (厚度面)
            thin = [
                [p[0], p[1], p[5], p[4]],   # 前面 y=0
                [p[2], p[3], p[7], p[6]],   # 后面 y=cY
                [p[0], p[3], p[7], p[4]],   # 左面 x=x0
                [p[1], p[2], p[6], p[5]],   # 右面 x=x0+cX
            ]

            poly_large = Poly3DCollection(large, alpha=0.72, linewidth=0.6,
                                          edgecolor='#333', facecolor=face_color)
            ax.add_collection3d(poly_large)
            poly_thin = Poly3DCollection(thin, alpha=0.82, linewidth=0.6,
                                         edgecolor='#333', facecolor=face_color)
            ax.add_collection3d(poly_thin)

            # ── 防爆阀: XZ 侧边 (174×71.7) 前立面 Y=0 的顶部 ──
            vx_center = x0 + cX * 0.55   # 偏右侧
            vz_center = z0 + cZ * 0.65   # 偏上方
            vy_face = y0                  # Y=0 前立面

            valve_r = 0.009
            valve_protrude = 0.006
            segs = 16
            theta = np.linspace(0, 2*np.pi, segs+1)
            th_x = vx_center + valve_r * np.cos(theta)
            th_z = vz_center + valve_r * np.sin(theta)
            vy_inner = np.full_like(th_x, vy_face)
            vy_outer = np.full_like(th_x, vy_face - valve_protrude)

            for i in range(segs):
                ax.add_collection3d(Poly3DCollection(
                    [[(th_x[i],   vy_inner[i],  th_z[i]),
                      (th_x[i+1], vy_inner[i+1],th_z[i+1]),
                      (th_x[i+1], vy_outer[i+1],th_z[i+1]),
                      (th_x[i],   vy_outer[i],  th_z[i])]],
                    alpha=0.95, facecolor='#cc2222', edgecolor='#770000',
                    linewidth=0.5))
            for i in range(1, segs):
                ax.add_collection3d(Poly3DCollection(
                    [[(th_x[0], vy_outer[0], th_z[0]),
                      (th_x[i], vy_outer[i], th_z[i]),
                      (th_x[i+1], vy_outer[i+1], th_z[i+1])]],
                    alpha=1.0, facecolor='#dd3333', edgecolor='#770000',
                    linewidth=0.3))

            # 编号
            ax.text(x0 + cX/2, y0 + cY + 0.005, z0 + cZ/2,
                    f'bat{idx+1}', ha='center', va='center',
                    fontsize=8, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.07)
    cbar.set_label('温升 ΔT [K]', fontsize=10)

    ax.set_xlabel('X (宽度) [m]', fontsize=9)
    ax.set_ylabel('Y (深度) [m]', fontsize=9)
    ax.set_zlabel('Z (堆叠方向) [m]', fontsize=9)

    x_span = cX * n_rows + gap_X * (n_rows-1)
    z_span = cZ * n_cols + gap_Y * (n_cols-1)
    pad = 0.03
    ax.set_xlim(offset_X - pad, offset_X + x_span + pad)
    ax.set_ylim(-pad, cY + pad * 3)
    ax.set_zlim(-pad, z_span + pad * 3)

    ax.grid(True, alpha=0.3)
    for pn in ('xaxis', 'yaxis', 'zaxis'):
        getattr(ax, pn).pane.fill = False

    ax.legend(handles=[
        mpatches.Patch(color='#cc2222', label='防爆阀 (前立面 XZ)'),
        mpatches.Patch(color='#5a8ac9', alpha=0.8, label='电芯壳 (按ΔT着色)'),
    ], loc='upper left', fontsize=9, framealpha=0.95)

    Tmn, Tmx = T_rise.min(), T_rise.max()
    ax.set_title(f'8S 电池包 · 2×4 大面背靠背堆叠 · 末态温度场\n'
                 f'T_amb={T_amb:.0f}K  ΔT {Tmn:.1f}~{Tmx:.1f}K  '
                 f'(4 层竖叠, 内芯积热, 角芯散热)',
                 fontsize=12, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 0.95, 1])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()

# demo_8s_foam_thermal.py  ——  8S 串联 + 全泡棉热耦合仿真
#
# 物理布局 (按真实电芯面尺寸修正): 2 组背靠背(X向) × 4 项链(Y向)
# 电芯三向尺寸: 厚 X=71mm, 深 Y=210mm, 高 Z=170mm
#   - 防爆阀面 = 71×210 (底面, 水平)
#   - 大面(背靠背) = 170×210 : bat1↔bat5 等大面相贴, 另一大面朝外
#   - 薄面(项链)   = 71×170  : bat1↔bat2 等薄面相接
#   排列:
#     Y列: [bat1]─泡─[bat2]─泡─[bat3]─泡─[bat4]   (沿深Y, 薄面项链)
#            │泡(大面)      │泡          │泡          │泡
#     Y列: [bat5]─泡─[bat6]─泡─[bat7]─泡─[bat8]
#   即 bat1 与 bat5 沿厚X背靠背(170×210大面), bat1 与 bat2 沿深Y项链(71×170薄面)
#
# 泡棉接触面 (10 条):
#   - 背靠背大面(X向): 4 条, A_large = 170×210 = 0.0357 m²
#   - 项链薄面(Y向):   6 条, A_thin  = 71×170  = 0.0121 m²
#   - R_th = d/k, d≈1.3mm EVA, k≈0.035 W/(m·K)
#
# 电气: 8S 串联, 总负载 200A 恒流放电。
# 冷却: 标定散热 — 端芯 h_outer=200, 中芯 h_inner=70 W/(m²·K)（按实验 2h 数据反演）
# 热模型: ThermalNetwork(8 节点) + 10 条 interface_resistance;
#         CellThermalModel 可视化最热电芯三维温度场。
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CJK = None
for _cand in [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]:
    if os.path.exists(_cand):
        _CJK = _cand
        break
if _CJK:
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ecm_pack as ep
from ecm_pack.thermal3d import CellThermalModel


def main():
    # ────── 标定参数覆盖（热模型标定用）──────
    import argparse as _ap_mod
    _pa = _ap_mod.ArgumentParser()
    _pa.add_argument("--h-inner", type=float, default=70.0)
    _pa.add_argument("--h-outer", type=float, default=200.0)
    _pa.add_argument("--k-foam", type=float, default=0.14)
    _pa.add_argument("--d-foam", type=float, default=0.0010)
    _cal = _pa.parse_known_args()[0]

    # ────── 工况参数（对齐真实测试条件）──────
    n_cells = 8
    I_load = 200.0               # 总负载 [A]（用户确认：200A）
    t_total = 7200.0             # 2 小时（约2h后电压截止自然停机）
    dt = 2.0
    n_steps = int(t_total / dt)  # 3600 步

    # ────── 泡棉参数 ──────
    k_foam = _cal.k_foam        # 压缩态 EVA 泡棉 [W/(m·K)]
    d_foam = _cal.d_foam        # 压缩后泡棉厚度
    R_th_foam = d_foam / k_foam
    # 大面 (170×210): bat1-bat5 背靠背 — 4 条
    A_large = 0.170 * 0.210       # 0.03570 m²
    # 薄面 (71×170): bat1-bat2 项链 — 6 条
    A_thin = 0.071 * 0.170        # 0.01207 m²
    G_large = A_large / R_th_foam
    G_thin = A_thin / R_th_foam
    print(f"泡棉: d≈{d_foam*1000:.1f}mm, k≈{k_foam}W/mK → R_th={R_th_foam:.4f} K·m²/W "
          f"({R_th_foam*1e4:.0f} K·cm²/W)")
    print(f"  背靠背大面 bat1↔bat5 A={A_large*1e4:.0f}cm² → G={G_large:.3f} W/K (4条)")
    print(f"  项链薄面 bat1↔bat2 A={A_thin*1e4:.0f}cm² → G={G_thin:.3f} W/K (6条)")

    # ────── 8 颗 314Ah 电芯 ──────
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
             for _ in range(n_cells)]

    # ────── 电气: 8S 串联 ──────
    nl, _, _ = ep.setup_circuit(8, 1)

    # ────── 热网络 ──────
    C_th_cell = 2300.0 * 1000.0 * (0.071 * 0.170 * 0.210)  # 71×170×210 → 5828 J/K
    h_conv = _cal.h_inner   # 内芯等效 h（模组内部温和换热）[W/(m²·K)]

    # 每芯暴露面积 — 修正堆叠: X=厚71mm(背靠背), Y=深210mm(项链), Z=高170mm
    # 面面积: 大面170×210=0.03570, 薄面71×170=0.01207, 阀面71×210=0.01491
    Al = 0.170 * 0.210    # 大面 170×210 (背靠背面, 外侧朝模组壳)
    At = 0.071 * 0.170     # 薄面 71×170 (项链面)
    A_valve = 0.071 * 0.210  # 顶/底面 (防爆阀面, 水平)
    # 端芯(bat1,4,5,8): 外侧大面 + 项链端薄面 + 顶 + 底 — 4 暴露面
    A_exp_corner = Al + At + 2 * A_valve   # 0.07759 m²
    # 中芯(bat2,3,6,7): 外侧大面 + 顶 + 底 — 3 暴露面 (薄面被左右邻居覆盖)
    A_exp_edge = Al + 2 * A_valve          # 0.06552 m²

    h_per_cell = np.zeros(n_cells)
    h_inner = h_conv
    h_outer = _cal.h_outer      # 端芯直接见模块壳
    h_per_cell[[0, 3, 4, 7]] = h_outer * A_exp_corner     # 端芯
    h_per_cell[[1, 2, 5, 6]] = h_inner * A_exp_edge       # 中芯

    print(f"\n  端芯暴露面={A_exp_corner*1e4:.0f}cm² h·A={h_per_cell[0]:.3f}W/K "
          f"(bat1,4,5,8)")
    print(f"  中芯暴露面={A_exp_edge*1e4:.0f}cm² h·A={h_per_cell[1]:.3f}W/K "
          f"(bat2,3,6,7)")

    # 10 条泡棉: 6 条同行厚度面 (A_thin) + 4 条跨行大面 (A_large)
    iface = [
        (0, 1, R_th_foam, A_thin),  (1, 2, R_th_foam, A_thin),
        (2, 3, R_th_foam, A_thin),  # row0 X方向
        (4, 5, R_th_foam, A_thin),  (5, 6, R_th_foam, A_thin),
        (6, 7, R_th_foam, A_thin),  # row1 X方向
        (0, 4, R_th_foam, A_large), (1, 5, R_th_foam, A_large),
        (2, 6, R_th_foam, A_large), (3, 7, R_th_foam, A_large),  # Y方向
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
    print(f" 8S + 全泡棉(10面)  200A 2h  大面背靠背+厚度项链  h_outer={h_outer}")
    print("=" * 65)
    for idx in range(n_cells):
        role = "角" if idx in [0, 3, 4, 7] else "边"
        print(f"  bat{idx+1} ({role})  "
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
        role = "角" if idx in [0,3,4,7] else "边"
        lbl = f"bat{idx+1}({role})"
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
    ax4.set_title(f"2×4 大面背靠背+厚度项链  泡棉R_th={R_th_foam*1e4:.0f} K·cm²/W  "
                  f"G_large={G_large:.1f}W/K G_thin={G_thin:.2f}W/K", fontsize=11)
    ax4.text(1.5, 1.65, "─ 项链Y向 (薄面71×170 A=121cm²)  ── 背靠背X向 (大面170×210 A=357cm²)",
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
        Lx=0.071, Ly=0.210, Lz=0.170, dim=3,
        nx=5, ny=8, nz=6,
        rho=2300.0, cp=1000.0, k=(0.7, 12.0, 11.6),
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
    print(f"\n  端芯平均温升: {rise_corner[-1]:.3f}K  中芯平均温升: {rise_edge[-1]:.3f}K")
    print(f"  端-中温差: {rise_edge[-1]-rise_corner[-1]:.4f}K "
          f"(中芯暴露面少→热堆积)")

    # ────── 3D 电池包可视化（2×4 排布 + 防爆阀）──────
    png5 = os.path.join(out_dir, "ecm_pack_8s_3dpack.png")
    _draw_pack_3d(T_cells[-1, :], png5,
                  cell_L=0.071, cell_W=0.210, cell_H=0.170,
                  gap_X=0.001, gap_Y=0.001, T_amb=298.15)
    print(f"  3D电池包: {png5}")



def _draw_pack_3d(T_vals, save_path, cell_L=0.071, cell_W=0.210, cell_H=0.170,
                  gap_X=0.001, gap_Y=0.001, T_amb=298.15):
    """8S电池包: 厚X=71(背靠背) × 深Y=210(项链) × 高Z=170, 防爆阀在顶面71×210。

    X: 2组 (bat1↔bat5 大面170×210背靠背, 间距71+泡棉)
    Y: 4项链 (bat1↔bat2 薄面71×170相接, 间距210+泡棉)
    Z: 170mm 高 → 防爆阀在顶面71×210
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.patches as mpatches

    T = np.asarray(T_vals, float).ravel()
    nX, nY = 2, 4   # 2组背靠背, 4项链
    T_rise = T - T_amb

    cX, cY, cZ = 0.071, 0.210, 0.170  # 厚71(背靠背X), 深210(项链Y), 高170(Z)
    pitch_X = cX + gap_X  # 71+1
    pitch_Y = cY + gap_Y  # 210+1

    fig = plt.figure(figsize=(15, 8), dpi=140)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=max(T_rise.min(), 0), vmax=T_rise.max() + 0.1)
    ax.view_init(elev=28, azim=-45)

    offset_X = -((nX-1) * pitch_X + cX) / 2
    offset_Y = -((nY-1) * pitch_Y + cY) / 2

    # 预计算所有电芯位置/颜色
    cells = []
    for ix in range(nX):
        for iy in range(nY):
            idx = ix * nY + iy  # bat1(0,0)=0, bat2(0,1)=1, bat3(0,2)=2, bat4(0,3)=3, bat5(1,0)=4, ...
            x0 = offset_X + ix * pitch_X
            y0 = offset_Y + iy * pitch_Y
            z0 = 0.0
            tr = T_rise[idx]
            face_color = cmap(norm(tr))
            p = np.array([
                [x0,       y0,       z0],
                [x0 + cX,  y0,       z0],
                [x0 + cX,  y0 + cY,  z0],
                [x0,       y0 + cY,  z0],
                [x0,       y0,       z0 + cZ],
                [x0 + cX,  y0,       z0 + cZ],
                [x0 + cX,  y0 + cY,  z0 + cZ],
                [x0,       y0 + cY,  z0 + cZ],
            ])
            # 6面: 底面XY(71×210), 顶面XY(71×210,防爆阀), 大面XZ(71×170,背靠背), 薄面YZ(210×170,项链)
            faces = [
                [p[0],p[1],p[2],p[3]],  [p[4],p[5],p[6],p[7]],
                [p[0],p[1],p[5],p[4]],  [p[2],p[3],p[7],p[6]],
                [p[0],p[3],p[7],p[4]],  [p[1],p[2],p[6],p[5]],
            ]
            cells.append((p, faces, face_color, x0, y0, z0, idx))

    # 第一遍: 仅画电芯主体 — 全部在最底层, 后续不会被阀/标签遮挡
    for (p, faces, face_color, x0, y0, z0, idx) in cells:
        poly = Poly3DCollection(faces, alpha=0.80, linewidth=0.5,
                                edgecolor='#333', facecolor=face_color)
        ax.add_collection3d(poly)

    # 防爆阀与标签改用 2D 顶层叠加 (见下方 savefig 前), 避免被电芯主体遮挡

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.07)
    cbar.set_label('温升 ΔT [K]', fontsize=10)

    ax.set_xlabel('X (2组大面背靠背 71+泡棉) [m]', fontsize=9)
    ax.set_ylabel('Y (4项链薄面相接 210+泡棉) [m]', fontsize=9)
    ax.set_zlabel('Z (高170mm 防爆阀↑顶面) [m]', fontsize=9)

    x_span = nX*cX + (nX-1)*gap_X; y_span = nY*cY + (nY-1)*gap_Y; pad = 0.03
    ax.set_xlim(offset_X-pad, offset_X+x_span+pad)
    ax.set_ylim(offset_Y-pad, offset_Y+y_span+pad)
    ax.set_zlim(-pad, cZ+0.05)
    # 统一 xyz 物理刻度: 1 数据单位 = 1 屏幕单位, 保留电芯真实 71×210×170 比例
    ax.set_box_aspect((x_span + 2*pad, y_span + 2*pad, (cZ + 0.05) + pad))

    ax.grid(True, alpha=0.3)
    for pn in ('xaxis','yaxis','zaxis'):
        getattr(ax, pn).pane.fill = False
    ax.legend(handles=[
        plt.Line2D([0],[0],marker='o',color='#cc2222',markersize=8,label='防爆阀(顶面)',lw=0),
        plt.Line2D([0],[0],marker='s',color='#5a8ac9',markersize=10,label='电芯壳(按ΔT着色)',lw=0),
    ], loc='upper left', fontsize=9, framealpha=0.95)

    Tmn, Tmx = T_rise.min(), T_rise.max()
    ax.set_title(f'8S电池包 · 2组大面背靠背 × 4项链薄面相接 · 防爆阀↑顶面'
                 f'  (T_amb={T_amb:.0f}K ΔT{Tmn:.1f}~{Tmx:.1f}K)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,0.95,1])

    # ── 2D 顶层叠加: 防爆阀(红点) + bat 标签 ──
    # 先 draw 锁定 renderer/dpi 与坐标轴位置, 再把每个电芯顶面中心投影到屏幕像素,
    # 用 fig.text 强制画在所有 3D 实体之上 (zorder=999/1000),
    # 彻底避免 matplotlib 3D 深度排序把后排阀/标签压到电芯底下。
    fig.canvas.draw()
    W = fig.get_figwidth() * fig.dpi
    H = fig.get_figheight() * fig.dpi
    for (p, faces, face_color, x0, y0, z0, idx) in cells:
        vx, vy = x0 + cX/2, y0 + cY/2
        vz = z0 + cZ
        a, b, _ = proj3d._scale_proj_transform(
            np.array([vx]), np.array([vy]), np.array([vz]), ax)
        px, py = ax.transData.transform((float(a[0]), float(b[0])))
        fx, fy = px / W, py / H
        # 防爆阀: 红点
        fig.text(fx, fy, '●', color='#cc2222', fontsize=20,
                 ha='center', va='center', zorder=999)
        # bat 标签: 白底黑字, 略高于阀
        fig.text(fx, fy + 0.013, f'bat{idx+1}',
                 color='black', fontsize=9, fontweight='bold',
                 ha='center', va='bottom', zorder=1000,
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

    fig.savefig(save_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()


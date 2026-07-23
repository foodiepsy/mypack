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
    k_foam = 0.035               # 压缩态 EVA 泡棉 [W/(m·K)]
    d_foam = 0.0013              # 压缩后 ~1.3mm
    R_th_foam = d_foam / k_foam
    # 大面 (174×207): bat1-bat5 跨行背靠背 — 4 条
    A_large = 0.174 * 0.207       # 0.03602 m²
    # 厚度面 (71.7×207): bat1-bat2 同行项链 — 6 条
    A_thin = 0.0717 * 0.207       # 0.01484 m²
    G_large = A_large / R_th_foam
    G_thin = A_thin / R_th_foam
    print(f"泡棉: d≈{d_foam*1000:.1f}mm, k≈{k_foam}W/mK → R_th={R_th_foam:.4f} K·m²/W "
          f"({R_th_foam*1e4:.0f} K·cm²/W)")
    print(f"  跨行大面 bat1↔bat5 A={A_large*1e4:.0f}cm² → G={G_large:.3f} W/K (4条)")
    print(f"  同行厚度 bat1↔bat2 A={A_thin*1e4:.0f}cm² → G={G_thin:.3f} W/K (6条)")

    # ────── 8 颗 314Ah 电芯 ──────
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
             for _ in range(n_cells)]

    # ────── 电气: 8S 串联 ──────
    nl, _, _ = ep.setup_circuit(8, 1)

    # ────── 热网络 ──────
    C_th_cell = 2300.0 * 1000.0 * (0.174 * 0.0717 * 0.207)  # 5941 J/K
    h_conv = 6.0   # 内芯等效 h（模组内部温和换热）[W/(m²·K)]

    # 每芯暴露面积 — 正确堆叠: X=4芯(174mm宽), Y=2芯(71.7mm深), Z=207mm高
    # 面面积: A_thin_YZ=0.01484, A_large_XZ=0.03602, A_top_XY=0.01248
    At = 0.0717*0.207    # 厚度面 71.7×207 (bat1↔bat2的项链面)
    Al = 0.174*0.207     # 大面 174×207 (bat1↔bat5的背靠背面)
    A_top = 0.174*0.0717 # 顶/底面 (防爆阀所在面)
    # 角芯(bat1,4,5,8): x端+ y端+ 顶+底 — 4 暴露面
    A_exp_corner = At + Al + 2*A_top  # 0.07582 m²
    # 边芯(bat2,3,6,7): y端+顶+底 — 3 暴露面 (x面被左右邻居覆盖)
    A_exp_edge = Al + 2*A_top          # 0.06096 m²

    h_per_cell = np.zeros(n_cells)
    h_inner = h_conv
    h_outer = 30.0              # 角芯直接见模块壳
    h_per_cell[[0, 3, 4, 7]] = h_outer * A_exp_corner     # 角芯
    h_per_cell[[1, 2, 5, 6]] = h_inner * A_exp_edge       # 边芯

    print(f"\n  角芯暴露面={A_exp_corner*1e4:.0f}cm² h·A={h_per_cell[0]:.3f}W/K "
          f"(bat1,4,5,8)")
    print(f"  边芯暴露面={A_exp_edge*1e4:.0f}cm² h·A={h_per_cell[1]:.3f}W/K "
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
    """8S电池包: 底面210×71坐XY, 防爆阀在上方210×71面。

    X: 4列 (bat1↔bat2厚度面71×174项链, 间距210+泡棉)
    Y: 2行 (bat1↔bat5大面174×210背靠背, 间距71+泡棉)
    Z: 174mm → 防爆阀在顶面210×71正中央
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patches as mpatches

    T = np.asarray(T_vals, float).ravel()
    nX, nY = 4, 2   # 4列项链, 2行背靠背
    T_rise = T - T_amb

    cX, cY, cZ = 0.210, 0.071, 0.174  # 底面 210×71 坐 XY, Z=174 高
    pitch_X = cX + gap_X  # 210+1
    pitch_Y = cY + gap_Y  # 71+1

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=max(T_rise.min(), 0), vmax=T_rise.max() + 0.1)
    ax.view_init(elev=28, azim=-45)

    offset_X = -((nX-1) * pitch_X + cX) / 2
    offset_Y = -((nY-1) * pitch_Y + cY) / 2

    for ix in range(nX):
        for iy in range(nY):
            idx = iy * nX + ix  # bat1(0,0)=0, bat5(1,0)=4, bat2(0,1)=1

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

            # 6面: 底面XY(210×71), 顶面XY(防爆阀), 大面XZ(210×174), 厚度面YZ(71×174)
            faces = [
                [p[0],p[1],p[2],p[3]],  [p[4],p[5],p[6],p[7]],
                [p[0],p[1],p[5],p[4]],  [p[2],p[3],p[7],p[6]],
                [p[0],p[3],p[7],p[4]],  [p[1],p[2],p[6],p[5]],
            ]
            poly = Poly3DCollection(faces, alpha=0.80, linewidth=0.5,
                                    edgecolor='#333', facecolor=face_color)
            ax.add_collection3d(poly)

            # 防爆阀: 顶面(Z=cZ)正中央, 圆柱凸起
            vx, vy = x0 + cX/2, y0 + cY/2
            vz = z0 + cZ
            r, h, s = 0.011, 0.006, 18
            th = np.linspace(0, 2*np.pi, s+1)
            tx, ty = vx + r*np.cos(th), vy + r*np.sin(th)
            zb, zt = np.full_like(tx, vz), np.full_like(tx, vz+h)
            for i in range(s):
                ax.add_collection3d(Poly3DCollection(
                    [[(tx[i],ty[i],zb[i]),(tx[i+1],ty[i+1],zb[i+1]),
                      (tx[i+1],ty[i+1],zt[i+1]),(tx[i],ty[i],zt[i])]],
                    alpha=0.95, facecolor='#cc2222', edgecolor='#770000', lw=0.4))
            for i in range(1, s):
                ax.add_collection3d(Poly3DCollection(
                    [[(tx[0],ty[0],zt[0]),(tx[i],ty[i],zt[i]),
                      (tx[i+1],ty[i+1],zt[i+1])]],
                    alpha=1.0, facecolor='#dd3333', edgecolor='#770000', lw=0.3))
            ax.text(vx, vy, vz+h+0.004, f'bat{idx+1}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.07)
    cbar.set_label('温升 ΔT [K]', fontsize=10)

    ax.set_xlabel('X (4列厚度项链 210+泡棉) [m]', fontsize=9)
    ax.set_ylabel('Y (2行大面背靠背 71+泡棉) [m]', fontsize=9)
    ax.set_zlabel('Z (高174mm 防爆阀↑) [m]', fontsize=9)

    x_span = nX*cX + (nX-1)*gap_X; y_span = nY*cY + (nY-1)*gap_Y; pad = 0.03
    ax.set_xlim(offset_X-pad, offset_X+x_span+pad)
    ax.set_ylim(offset_Y-pad, offset_Y+y_span+pad)
    ax.set_zlim(-pad, cZ+0.05)

    ax.grid(True, alpha=0.3)
    for pn in ('xaxis','yaxis','zaxis'):
        getattr(ax, pn).pane.fill = False
    ax.legend(handles=[
        plt.Line2D([0],[0],marker='o',color='#cc2222',markersize=8,label='防爆阀(顶面)',lw=0),
        plt.Line2D([0],[0],marker='s',color='#5a8ac9',markersize=10,label='电芯壳(按ΔT着色)',lw=0),
    ], loc='upper left', fontsize=9, framealpha=0.95)

    Tmn, Tmx = T_rise.min(), T_rise.max()
    ax.set_title(f'8S电池包 · 4列厚度项链 × 2行大面背靠背 · 防爆阀↑'
                 f'  (T_amb={T_amb:.0f}K ΔT{Tmn:.1f}~{Tmx:.1f}K)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,0.95,1])
    fig.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


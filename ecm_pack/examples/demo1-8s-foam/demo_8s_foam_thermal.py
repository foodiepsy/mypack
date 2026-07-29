# demo_8s_foam_thermal.py  ——  8S 大面背靠背串联 + 薄侧泡棉(不对称) + 三维非对称冷却仿真
#
# 物理布局（2026-07-28 用户修正）：
#   环境: 25°C 强制自然对流
#   8 颗 314Ah 电芯沿厚度方向(Y) 大面背靠背 1×8 串联，电芯间直接贴合(无泡棉)。
#   薄侧: +X面、+Z面 = 泡棉(k=0.04,1mm) + 25°C空气强制对流；
#         -X面、-Z面 = 直接25°C空气强制对流（无泡棉隔层）。
#   顶部: 塑料片(k=0.2,d=0.2mm) + 25°C强制对流。
#   底部(大面尾部): 25°C空气强制对流(非绝热)。
#   无冷板。
#
# 三维热模型 StackThermal3D(thermal3d_stack.py):
#   foam_faces=["x0","z0"]  — +X/+Z贴泡棉, -X/-Z直接空气
#   k_top=0.2, d_top=2e-4   — 顶部塑料薄层串联对流
#   h_top=h_side=50         — 25°C强制对流, T_amb=298.15K
#   h_bottom=50              — 底部大面=空气强制对流(非绝热)
#
# 电气: 8S 串联, 总负载 157A 恒流放电（0.5C, 2h）。
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import ecm_pack as ep
from ecm_pack.thermal3d_stack import StackThermal3D

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")


def main():
    # ---- 工况参数 ----
    n_cells = 8
    I_load = 157.0
    t_total = 7200.0
    dt = 2.0
    n_steps = int(t_total / dt)

    # ---- 薄侧泡棉（仅 +X/+Z 面）----
    k_foam = 0.04
    d_foam = 0.001
    h_env = 50.0                      # 25°C 强制自然对流 [W/(m^2·K)]
    T_amb = 298.15                    # 环境温度

    # ---- 顶部塑料片（薄层，导热率 > 泡棉）----
    k_plastic = 0.2                   # PET 塑料片 [W/(m·K)]
    d_plastic = 0.0002                # 0.2 mm

    Lx, Ly, Lz = 0.174, 0.0717, 0.207

    # ---- 薄侧面积与等效导热（不对称）----
    # 贴泡棉面(+X,+Z): 面积 A_foam = n_cells*Ly*(Lx+Lz)
    # 直接空气面(-X,-Z): 面积 A_bare = n_cells*Ly*(Lx+Lz) (相等)
    A_foam = n_cells * Ly * (Lx + Lz)
    A_bare = A_foam
    Rpp_foam = d_foam / k_foam              # 泡棉单位面积热阻 = 0.025
    Rpp_conv = 1.0 / h_env                   # 对流单位面积热阻 = 0.02
    G_foam_side = A_foam / (Rpp_foam + Rpp_conv)
    G_bare_side = A_bare * h_env             # 直接空气面无泡棉
    Rpp_plastic = d_plastic / k_plastic      # 塑料片单位面积热阻 ~ 0.0010
    G_top = (Lx * Lz) / (Rpp_plastic + Rpp_conv)
    G_bottom = Lx * Lz * h_env             # 底部大面=空气对流(非绝热)

    print(f"薄侧(+X,+Z): 1mm EVA k={k_foam} 面积={A_foam*1e4:.0f}cm^2 -> G={G_foam_side:.2f} W/K")
    print(f"薄侧(-X,-Z): 直接空气 h={h_env} 面积={A_bare*1e4:.0f}cm^2 -> G={G_bare_side:.2f} W/K")
    print(f"顶部塑料: k={k_plastic} d={d_plastic*1e3:.1f}mm -> G_top={G_top:.2f} W/K")
    print(f"底部: 空气对流 h={h_env}  (非绝热)  G={G_bottom:.2f}")

    # ---- 8 颗 314Ah 电芯 ----
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
             for _ in range(n_cells)]
    spec = cells[0].spec
    Lx, Ly, Lz = spec.Lx, spec.Ly, spec.Lz
    k_cell = spec.k
    rho, cp = spec.rho, spec.cp
    A_foam = n_cells * Ly * (Lx + Lz)
    G_foam_side = A_foam / (Rpp_foam + Rpp_conv)
    G_bare_side = A_foam * h_env
    G_top = (Lx * Lz) / (Rpp_plastic + Rpp_conv)
    print(f"  真实几何: Lx={Lx} Ly={Ly} Lz={Lz} -> 大面={Lx*Lz*1e4:.0f}cm^2")
    print(f"  重算薄侧泡棉G={G_foam_side:.2f} 直接空气G={G_bare_side:.2f} 顶塑料G={G_top:.2f}")

    # ---- 电气: 8S 串联 ----
    nl, _, _ = ep.setup_circuit(8, 1)

    # ---- 三维复合热模型 ----
    thermal = StackThermal3D(
        n_cells=n_cells, Lx=Lx, Ly=Ly, Lz=Lz,
        nx=4, ny=5, nz=8,
        cell_k=k_cell, cell_rho=rho, cell_cp=cp,
        foam_k=k_foam, foam_thickness=d_foam,
        foam_faces=["x0", "z0"],         # +X/+Z贴泡棉, -X/-Z直接空气
        k_top=k_plastic, d_top=d_plastic,  # 顶部塑料薄层
        h_top=h_env, T_top=T_amb,         # 顶部 25°C强制对流（无冷板）
        h_bottom=h_env, T_bottom=T_amb,    # 底部大面=25°C空气强制对流(非绝热)
        h_side=h_env, T_amb=T_amb,        # 薄侧 25°C强制对流
        T_init=T_amb,
    )
    print(f"\n三维热模型: 8芯沿Y直接贴合(无泡棉), 网格 {thermal.NX}x{thermal.NY}x{thermal.NZ} = {thermal.N}节点")
    print(f"  +X/+Z薄侧=泡棉+空气  -X/-Z薄侧=直接空气  顶=塑料片(k={k_plastic})+空气")

    # ---- Pack 耦合求解 ----
    pack = ep.Pack(cells, nl, thermal=thermal, v_cut_lower=2.0)
    out = pack.solve(dt=dt, control=I_load, control_type="current",
                     n_steps=n_steps, record_every=60)

    time_arr = out["Time [s]"]
    T_cells = out["Cell temperature [K]"]
    Vt = out["Pack terminal voltage [V]"]
    soc = out["Cell SoC"]

    # ---- 报告 ----
    T_end = T_cells[-1, :]
    # align the 3D thermal object's bulk temperatures with the reported final
    # snapshot (Pack may have continued past the last record before terminating)
    thermal.T = T_end.copy()
    print("\n" + "=" * 65)
    print(f" 8S 大面背靠背 1×8 贴合(无泡棉) / 薄侧不对称(+X,+Z泡棉/-X,-Z直接空气)")
    print(f" 25°C强制对流(h=50) 顶部塑料片 底部空气对流 0.5C=157A / 2h")
    print("=" * 65)
    for idx in range(n_cells):
        if idx == 0:
            ylab = "0(顶)"
        elif idx == n_cells - 1:
            ylab = "max(底空气)"
        else:
            ylab = str(idx)
        print(f"  bat{idx+1} (y={ylab})  T_final={T_end[idx]:.2f}K  "
              f"ΔT={T_end[idx]-T_end[0]:+.3f}K  SoC={soc[-1,idx]:.4f}")
    dT_pack = T_end.max() - T_end.min()
    print(f"  端电压末态: {Vt[-1]:.2f} V")
    print(f"  整包ΔT_max: {dT_pack:.4f}K  "
          f"最热(bat{int(np.argmax(T_end))+1}@{T_end.max():.2f}K)  "
          f"最凉(bat{int(np.argmin(T_end))+1}@{T_end.min():.2f}K)")
    print(f"  bat8-bat1={T_end[-1]-T_end[0]:+.3f}K（底部空气对流->bat8最凉，顶部塑料隔热->bat1略高，中间芯最热）")

    # ---- 图1: 整包温度 + 端电压 ----
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    for idx in range(n_cells):
        ax1.plot(time_arr/60, T_cells[:, idx], color=colors[idx], lw=1.2, label=f"bat{idx+1}")
    ax1.set(ylabel="温度 [K]",
            title=f"8S 大面背靠背贴 温度演变（薄侧不对称+顶塑料+底空气对流） dT={dT_pack:.3f}K")
    ax1.legend(ncol=4, fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.plot(time_arr/60, Vt, color="#1f77b4", lw=1.6)
    ax2.set(xlabel="时间 [min]", ylabel="端电压 [V]", title="整包端电压")
    ax2.grid(alpha=0.3)
    fig1.tight_layout()
    png1 = os.path.join(out_dir, "ecm_pack_8s_foam_temp.png")
    fig1.savefig(png1, dpi=130)
    plt.close(fig1)
    print(f"\n  温度曲线: {png1}")

    # ---- 图2: 末态温升 bar ----
    fig2, ax3 = plt.subplots(figsize=(10, 3.5))
    x_pos = np.arange(n_cells)
    rise = T_end - 298.15
    bars = ax3.bar(x_pos, rise, color=colors, edgecolor="gray", linewidth=0.8)
    ax3.set(xticks=x_pos, xticklabels=[f"bat{i+1}" for i in range(n_cells)],
            ylabel="温升 ΔT [K]",
            title=f"末态各芯温升 整包dT={dT_pack:.4f}K")
    for b, v in zip(bars, rise):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{v:.2f}", ha="center", fontsize=7.5)
    fig2.tight_layout()
    png2 = os.path.join(out_dir, "ecm_pack_8s_foam_bar.png")
    fig2.savefig(png2, dpi=130)
    plt.close(fig2)
    print(f"  温升柱图: {png2}")

    # ---- 图3: 堆叠示意图（大面贴合 + 不对称薄侧泡棉 + 顶塑料）----
    fig3, ax4 = plt.subplots(figsize=(11, 3.5))
    x0 = 0.0; cw = 1.0
    for idx in range(n_cells):
        x = x0 + idx * cw
        fc = "lightblue" if idx in (0, n_cells-1) else "lightcyan"
        ax4.add_patch(plt.Rectangle((x, 0.5), cw, 1.0, fc=fc, ec="gray", lw=1.3))
        ax4.text(x+cw/2, 1.0, f"bat{idx+1}", ha="center", va="center", fontsize=8)
        if idx < n_cells - 1:
            ax4.text(x+cw, 1.0, "贴合", ha="center", va="center", fontsize=6, color="gray")
    # +X/+Z 薄侧 = 泡棉 -> 空气（画在电芯上方）
    for idx in range(n_cells):
        x = x0 + idx * cw
        ax4.add_patch(plt.Rectangle((x, 1.50), cw, 0.06, fc="orange", ec="none", alpha=0.7))
    ax4.text(x0 + n_cells*cw/2, 1.72, "+X/+Z薄侧: 泡棉 -> 空气(强制对流 h=50)",
             ha="center", fontsize=9, color="darkorange")
    # -X/-Z 薄侧 = 直接空气（画在电芯下方）
    for idx in range(n_cells):
        x = x0 + idx * cw
        ax4.add_patch(plt.Rectangle((x, 0.42), cw, 0.08, fc="lightblue", ec="none", alpha=0.5))
    ax4.text(x0 + n_cells*cw/2, 0.30, "-X/-Z薄侧: 直接空气(强制对流 h=50)",
             ha="center", fontsize=9, color="steelblue")
    # 顶部塑料片（左端外 = bat1 顶面）
    ax4.add_patch(plt.Rectangle((x0-0.30, 0.5), 0.2, 1.0, fc="lightgreen", ec="green", lw=1.2))
    ax4.text(x0-0.20, 1.85, "顶塑料片\nk=0.2\n+空气", ha="center", fontsize=7, color="green")
    # 底部空气对流（右端外 = bat8 底面，非绝热）
    ax4.add_patch(plt.Rectangle((x0+n_cells*cw-0.10, 0.5), 0.1, 1.0, fc="lightblue", ec="navy", lw=1.2))
    ax4.text(x0+n_cells*cw, 1.85, "底部空气\n对流", ha="center", fontsize=8, color="navy")
    ax4.set(xlim=(x0-0.5, x0+n_cells*cw+0.5), ylim=(0.1, 2.1), xticks=[], yticks=[])
    ax4.set_title("1×8 大面背靠背直接贴合(无泡棉)+薄侧不对称+顶塑料片+底空气对流", fontsize=11)
    fig3.tight_layout()
    png3 = os.path.join(out_dir, "ecm_pack_8s_foam_network.png")
    fig3.savefig(png3, dpi=130)
    plt.close(fig3)
    print(f"  堆叠示意: {png3}")

    # ---- 图4: 沿堆叠轴(Y) 三维温度剖面 ----
    fig4, ax5 = plt.subplots(figsize=(11, 4))
    thermal.plot_y_profile(ax=ax5)
    fig4.tight_layout()
    png4 = os.path.join(out_dir, "ecm_pack_8s_foam_profile.png")
    fig4.savefig(png4, dpi=130)
    plt.close(fig4)
    print(f"  堆叠剖面: {png4}")

    # ---- 图5: X–Y 三维温度场切片 ----
    fig5, ax6 = plt.subplots(figsize=(11, 4))
    thermal.plot_xz_slice(z_frac=0.5, ax=ax6)
    fig5.tight_layout()
    png5 = os.path.join(out_dir, "ecm_pack_8s_foam_3dcell.png")
    fig5.savefig(png5, dpi=130)
    plt.close(fig5)
    print(f"  三维切片: {png5}")

    # ---- 图6: 真 3D 包体渲染（8 个体块 + 泡棉/空气 + 顶塑料 + 底空气对流）----
    png6 = os.path.join(out_dir, "ecm_pack_8s_3dpack.png")
    fig6, _ = thermal.plot_3d_pack(save_path=png6, dpi=130)
    plt.close(fig6)
    print(f"  三维包体: {png6}")

    # ---- CSV ----
    csv_path = os.path.join(out_dir, "ecm_pack_8s_foam_data.csv")
    header = "Time_s,Vt_V," + ",".join([f"T_bat{i+1}_K" for i in range(n_cells)]) \
             + "," + ",".join([f"SOC_bat{i+1}" for i in range(n_cells)])
    data_cols = [np.asarray(time_arr, float).flatten(), np.asarray(Vt, float).flatten()]
    for i in range(n_cells):
        data_cols.append(np.asarray(T_cells[:, i], float).flatten())
    for i in range(n_cells):
        data_cols.append(np.asarray(soc[:, i], float).flatten())
    np.savetxt(csv_path, np.column_stack(data_cols), delimiter=",",
               header=header, fmt="%.6f")
    print(f"  数据: {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

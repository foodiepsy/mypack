# demo_8s_foam_thermal.py  ——  8S 大面背靠背串联 + 全泡棉 + 三维非对称冷却仿真
#
# 物理布局: 8 颗 314Ah 电芯沿厚度方向(Y) 大面背靠背排成 1×8 串联，
#           相邻大面（XZ 面, 0.174×0.207=0.036 m?）之间夹 1mm EVA/PE 泡棉，共 7 个界面。
#           即「电芯—泡棉—电芯—泡棉—…—电芯」的一条堆叠链。
#
# 三维热模型 StackThermal3D(thermal3d_stack.py):
#   把 8 颗芯 + 7 层泡棉建成一个三维复合有限体积域（沿 Y 堆叠），
#   电芯各向异性导热 k=(12,0.7,11.6)，泡棉 k=0.04，逐面非对称对流冷却：
#     - 顶部（bat1 外大面, 全局 y=0）：25°C 强制对流冷板  h_top=50 W/m?K, T=298.15K
#     - 底部（bat8 外大面, 全局 y=max）：绝热（无换热） h_bottom=0
#     - 侧面（x0/x1/z0/z1 小面）：强对流              h_side=50 W/m?K, T=298.15K
#
# 电气: 8S 串联, 总负载 314A 恒流放电（1C, 2h）。
# 热耦合: Pack 双步循环把 ECM 产热(含 R_contact 发热) 喂给 StackThermal3D。
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
from ecm_pack.thermal3d_stack import StackThermal3D

out_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # ────── 工况参数 ──────
    n_cells = 8
    I_load = 314.0               # 总负载 1C [A]（8S 串联，每芯 314A）
    t_total = 7200.0             # 2 小时
    dt = 2.0
    n_steps = int(t_total / dt)  # 3600 步

    # ────── 泡棉参数（相邻大面之间）──────
    k_foam = 0.04                # EVA/PE 泡棉 [W/(m·K)]
    d_foam = 0.001               # 1 mm
    R_th_foam = d_foam / k_foam  # 0.025 K·m?/W = 250 K·cm?/W
    A_row = 0.174 * 0.207        # 大面 XZ = 0.03602 m?（背靠背接触面）
    G_foam = A_row / R_th_foam   # ≈ 1.44 W/K（每界面等效导热）
    print(f"泡棉: 1mm EVA k={k_foam} → R_th={R_th_foam:.4f} K·m?/W ({R_th_foam*1e4:.0f} K·cm?/W)")
    print(f"  7 个背靠背大面界面, 每个 A={A_row*1e4:.0f}cm? → G={G_foam:.3f} W/K")

    # ────── 8 颗 314Ah 电芯 ──────
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
             for _ in range(n_cells)]
    spec = cells[0].spec
    Lx, Ly, Lz = spec.Lx, spec.Ly, spec.Lz
    k_cell = spec.k
    rho, cp = spec.rho, spec.cp

    # ────── 电气: 8S 串联 ──────
    nl, _, _ = ep.setup_circuit(8, 1)

    # ────── 三维复合热模型（大面背靠背 1×8 堆叠 + 逐面非对称冷却）──────
    #   bat1 外大面(y=0)     → 25°C 强制对流冷板
    #   bat8 外大面(y=max)   → 绝热
    #   侧面(x0/x1/z0/z1)    → 强对流
    thermal = StackThermal3D(
        n_cells=n_cells, Lx=Lx, Ly=Ly, Lz=Lz,
        nx=4, ny=5, nz=8,                 # 每芯网格
        cell_k=k_cell, cell_rho=rho, cell_cp=cp,
        foam_k=k_foam, foam_thickness=d_foam, foam_ny=1,
        h_top=50.0, T_top=298.15,         # 顶部 25°C 冷板（强制对流）
        h_bottom=0.0, T_bottom=298.15,    # 底部绝热（无换热）
        h_side=50.0, T_amb=298.15,        # 侧面强对流
        T_init=298.15,
    )
    print(f"\n三维热模型: 8 芯沿 Y 堆叠, 网格 {thermal.NX}×{thermal.NY}×{thermal.NZ}"
          f" = {thermal.N} 节点, 泡棉层 {n_cells-1}")

    # ────── Pack 耦合求解 ──────
    pack = ep.Pack(cells, nl, thermal=thermal, v_cut_lower=2.0)
    out = pack.solve(dt=dt, control=I_load, control_type="current",
                     n_steps=n_steps, record_every=60)

    time_arr = out["Time [s]"]
    T_cells = out["Cell temperature [K]"]
    Vt = out["Pack terminal voltage [V]"]
    I_cell = out["Cell current [A]"]
    soc = out["Cell SoC"]

    # ────── 报告 ──────
    T_end = T_cells[-1, :]
    print("\n" + "=" * 65)
    print(f" 8S 大面背靠背串联 / 1×8 堆叠 / 全泡棉 / 三维非对称冷却")
    print(f" 顶部25°C冷板(h=50)  底部绝热  侧面强对流(h=50)  1C=314A / 2h")
    print("=" * 65)
    for idx in range(n_cells):
        if idx == 0:
            ylab = "0(冷板端)"
        elif idx == n_cells - 1:
            ylab = "max(绝热端)"
        else:
            ylab = str(idx)
        print(f"  bat{idx+1} (y={ylab})  T_final={T_end[idx]:.2f}K  "
              f"ΔT={T_end[idx]-T_end[0]:+.3f}K  SoC={soc[-1,idx]:.4f}")
    dT_pack = T_end.max() - T_end.min()
    print(f"  端电压末态: {Vt[-1]:.2f} V")
    print(f"  整包ΔT_max: {dT_pack:.4f}K  "
          f"最热(bat{int(np.argmax(T_end))+1}@{T_end.max():.2f}K)  "
          f"最凉(bat{int(np.argmin(T_end))+1}@{T_end.min():.2f}K)")
    print(f"  → 冷板端bat1={T_end[0]:.2f}K, 绝热端bat8={T_end[-1]:.2f}K, "
          f"bat8-bat1={T_end[-1]-T_end[0]:+.3f}K（应为正：底部绝热更热）")

    # ────── 图1: 整包温度 + 端电压 ──────
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    for idx in range(n_cells):
        ax1.plot(time_arr/60, T_cells[:, idx], color=colors[idx], lw=1.2, label=f"bat{idx+1}")
    ax1.set(ylabel="温度 [K]",
            title=f"8S 大面背靠背 1×8 堆叠温度演变（三维非对称冷却） 整包ΔT={dT_pack:.3f}K")
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

    # ────── 图2: 末态温升 bar ──────
    fig2, ax3 = plt.subplots(figsize=(10, 3.5))
    x_pos = np.arange(n_cells)
    rise = T_end - 298.15
    bars = ax3.bar(x_pos, rise, color=colors, edgecolor="gray", linewidth=0.8)
    ax3.set(xticks=x_pos, xticklabels=[f"bat{i+1}" for i in range(n_cells)],
            ylabel="温升 ΔT [K]",
            title=f"末态各芯温升 整包ΔT={dT_pack:.4f}K （底部绝热→bat8最热, 顶部冷板→bat1最凉）")
    for b, v in zip(bars, rise):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{v:.2f}", ha="center", fontsize=7.5)
    ax3.annotate("", xy=(0, rise[0]+0.05), xytext=(n_cells-1, rise[-1]+0.05),
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax3.text((n_cells-1)/2, rise.max()+0.15, "沿堆叠轴: 冷板端→绝热端 温度递增",
             color="red", fontsize=9, ha="center")
    fig2.tight_layout()
    png2 = os.path.join(out_dir, "ecm_pack_8s_foam_bar.png")
    fig2.savefig(png2, dpi=130)
    plt.close(fig2)
    print(f"  温升柱图: {png2}")

    # ────── 图3: 堆叠示意图（链 + 冷板/绝热）──────
    fig3, ax4 = plt.subplots(figsize=(11, 3))
    x0 = 0.0
    cw = 1.0   # 电芯框宽
    fw = 0.18  # 泡棉间隙
    for idx in range(n_cells):
        x = x0 + idx * (cw + fw)
        fc = "lightblue" if idx in (0, n_cells-1) else "lightcyan"
        ax4.add_patch(plt.Rectangle((x, 0.3), cw, 1.4, fc=fc, ec="gray", lw=1.3))
        ax4.text(x+cw/2, 1.0, f"bat{idx+1}", ha="center", va="center", fontsize=8)
        if idx < n_cells-1:
            ax4.add_patch(plt.Rectangle((x+cw, 0.7), fw, 0.6, fc="orange", ec="none", alpha=0.6))
            ax4.text(x+cw+fw/2, 1.55, "泡棉", ha="center", fontsize=6, color="darkorange")
    # 冷板（左）
    ax4.add_patch(plt.Rectangle((x0-0.35, 0.3), 0.3, 1.4, fc="skyblue", ec="blue", lw=1.5))
    ax4.text(x0-0.2, 2.0, "25°C\n冷板", ha="center", fontsize=8, color="blue")
    # 绝热（右）
    ax4.add_patch(plt.Rectangle((x0+n_cells*(cw+fw)-0.15, 0.3), 0.15, 1.4, fc="white", ec="gray", lw=1, ls="--"))
    ax4.text(x0+n_cells*(cw+fw), 2.0, "绝热", ha="center", fontsize=8, color="gray")
    ax4.set(xlim=(x0-0.6, x0+n_cells*(cw+fw)+0.6), ylim=(0, 2.4),
            xticks=[], yticks=[])
    ax4.set_title("1×8 大面背靠背串联堆叠：左端25°C冷板 / 右端绝热 / 侧面强对流",
                  fontsize=11)
    ax4.set_aspect("equal")
    fig3.tight_layout()
    png3 = os.path.join(out_dir, "ecm_pack_8s_foam_network.png")
    fig3.savefig(png3, dpi=130)
    plt.close(fig3)
    print(f"  堆叠示意: {png3}")

    # ────── 图4: 沿堆叠轴(Y) 三维温度剖面 ──────
    fig4, ax5 = plt.subplots(figsize=(11, 4))
    thermal.plot_y_profile(ax=ax5)
    fig4.tight_layout()
    png4 = os.path.join(out_dir, "ecm_pack_8s_foam_profile.png")
    fig4.savefig(png4, dpi=130)
    plt.close(fig4)
    print(f"  堆叠剖面: {png4}")

    # ────── 图5: X–Y 三维温度场切片 ──────
    fig5, ax6 = plt.subplots(figsize=(11, 4))
    thermal.plot_xz_slice(z_frac=0.5, ax=ax6)
    fig5.tight_layout()
    png5 = os.path.join(out_dir, "ecm_pack_8s_foam_3dcell.png")
    fig5.savefig(png5, dpi=130)
    plt.close(fig5)
    print(f"  三维切片: {png5}")

    # ────── CSV ──────
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

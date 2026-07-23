# demo_8s_foam_thermal.py  ——  8S 串联 + 芯间 1mm 泡棉热耦合仿真
#
# 物理布局: 2 行 × 4 芯, 每行内芯间夹 1mm EVA/PE 泡棉
#   Row1: [bat1]─泡─[bat2]─泡─[bat3]─泡─[bat4]
#   Row2: [bat5]─泡─[bat6]─泡─[bat7]─泡─[bat8]
# 电气: 8S 串联, 总负载 200A 恒流放电。
# 热: ThermalNetwork(8 节点) + interface_resistance 建模泡棉层间热阻;
#     结束后用 CellThermalModel 三维可视化最热电芯的温度场。
import sys, os, json
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
    # ────── 工况参数 ──────
    n_cells = 8
    I_load = 200.0               # 总负载 [A]
    t_total = 600.0              # 仿真时长 [s] (10 min)
    dt = 1.0
    n_steps = int(t_total / dt)

    # ────── 泡棉参数 ──────
    k_foam = 0.04                # EVA/PE 泡棉导热系数 [W/(m·K)]
    d_foam = 0.001               # 1 mm
    A_contact = 0.174 * 0.207    # X×Z 面接触面积 [m²]
    R_th_foam = d_foam / k_foam  # 热接触电阻 [K·m²/W]
    print(f"泡棉: d={d_foam*1000:.0f}mm, k={k_foam}W/mK → R_th={R_th_foam:.4f} K·m²/W "
          f"(≈ {R_th_foam*1e4:.1f} K·cm²/W)")
    print(f"接触面积 A={A_contact*1e4:.0f} cm² → 等效 G={A_contact/R_th_foam:.2f} W/K")

    # ────── 8 颗 314Ah 电芯 ──────
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
             for _ in range(n_cells)]

    # ────── 电气拓扑: 8S 串联 ──────
    nl, _, _ = ep.setup_circuit(8, 1)  # 8 串 1 并

    # ────── 热网络: interface_resistance 建模泡棉 ──────
    C_th_cell = 2300.0 * 1000.0 * (0.174 * 0.0717 * 0.207)  # ρ·cp·V ≈ 5941 J/K
    # 每芯对流: h_conv × 暴露面积
    # 暴露面 = 总体 - 被泡棉覆盖的面
    # 末芯(bat1,4,5,8): 仅一面被泡棉覆盖 → 暴露面 = 总体 - A_contact/2
    # 中芯(bat2,3,6,7): 两面被泡棉覆盖 → 暴露面 = 总体 - A_contact
    total_surf = 2 * (0.174*0.0717 + 0.174*0.207 + 0.0717*0.207)  # ≈ 0.1267 m²
    h_conv = 5.0  # 自然对流 [W/(m²·K)]
    h_per_cell = np.zeros(n_cells)
    # Row1: bat1(idx0), bat2(1), bat3(2), bat4(3)
    # Row2: bat5(idx4), bat6(5), bat7(6), bat8(7)
    end_cells = [0, 3, 4, 7]   # bat1, bat4, bat5, bat8
    mid_cells = [1, 2, 5, 6]   # bat2, bat3, bat6, bat7
    h_per_cell[end_cells] = h_conv * (total_surf - A_contact)  # ~0.274 W/K
    h_per_cell[mid_cells] = h_conv * (total_surf - 2*A_contact) # ~0.093 W/K
    h_per_cell = np.maximum(h_per_cell, 0.05)  # 最低也有少量对流

    # 泡棉层间热阻: (i, j, R_th, A)
    iface = [
        (0, 1, R_th_foam, A_contact),   # bat1-bat2
        (1, 2, R_th_foam, A_contact),   # bat2-bat3
        (2, 3, R_th_foam, A_contact),   # bat3-bat4
        (4, 5, R_th_foam, A_contact),   # bat5-bat6
        (5, 6, R_th_foam, A_contact),   # bat6-bat7
        (6, 7, R_th_foam, A_contact),   # bat7-bat8
    ]

    thermal = ep.ThermalNetwork(
        n_cells, C_th=C_th_cell, h=h_per_cell, T_amb=298.15,
        interface_resistance=iface, T_init=298.15,
    )

    # ────── Pack 耦合求解 ──────
    pack = ep.Pack(cells, nl, thermal=thermal, v_cut_lower=2.0)
    out = pack.solve(dt=dt, control=I_load, control_type="current",
                     n_steps=n_steps, record_every=10)

    t = out["Time [s]"]
    T_cells = out["Cell temperature [K]"]     # (n_rec, 8)
    Vt = out["Pack terminal voltage [V]"]
    I_cell = out["Cell current [A]"]
    soc = out["Cell SoC"]

    # ────── 报告 ──────
    print()
    print("=" * 60)
    print(f" 8S + 1mm 泡棉  200A / 10min  仿真结果")
    print("=" * 60)
    for idx in range(n_cells):
        row = "Row1" if idx < 4 else "Row2"
        pos = idx % 4 + 1
        print(f"  bat{idx+1} ({row}#{pos})  "
              f"T_init={T_cells[0,idx]:.2f}K  "
              f"T_final={T_cells[-1,idx]:.2f}K  "
              f"ΔT={T_cells[-1,idx]-T_cells[0,idx]:.3f}K  "
              f"SoC={soc[-1,idx]:.4f}")
    T_all_end = T_cells[-1, :]
    print(f"  端电压末态: {Vt[-1]:.2f}V")
    print(f"  整包ΔT_max: {T_all_end.max()-T_all_end.min():.3f}K  "
          f"(最热: bat{int(np.argmax(T_all_end))+1}@{T_all_end.max():.2f}K, "
          f"最凉: bat{int(np.argmin(T_all_end))+1}@{T_all_end.min():.2f}K)")

    # ────── 图 1: 整包温度 + 端电压 ──────
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    for idx in range(n_cells):
        lbl = f"bat{idx+1}" + ("(端)" if idx in [0,3,4,7] else "(中)")
        ax1.plot(t/60, T_cells[:, idx], color=colors[idx], lw=1.2, label=lbl)
    ax1.set(ylabel="温度 [K]", title="8S 电芯温度演变（200A 放电 10min）")
    ax1.legend(ncol=4, fontsize=7, loc="best")
    ax1.grid(alpha=0.3)
    ax2.plot(t/60, Vt, color="#1f77b4", lw=1.6)
    ax2.set(xlabel="时间 [min]", ylabel="端电压 [V]", title="整包端电压")
    ax2.grid(alpha=0.3)
    fig1.tight_layout()
    png1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "ecm_pack_8s_foam_temp.png")
    png1 = os.path.abspath(png1)
    fig1.savefig(png1, dpi=130)
    plt.close(fig1)
    print(f"\n  整包温度图: {png1}")

    # ────── 图 2: 末态温度分布 bar chart ──────
    fig2, ax3 = plt.subplots(figsize=(8, 3))
    x_pos = np.arange(n_cells)
    bars = ax3.bar(x_pos, T_all_end - 298.15, color=colors, edgecolor="gray")
    ax3.set(xticks=x_pos, xticklabels=[f"bat{i+1}" for i in range(n_cells)],
            ylabel="温升 ΔT [K]", title="末态各电芯温升 (t=10min)")
    for b, v in zip(bars, T_all_end - 298.15):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f"{v:.2f}", ha="center", fontsize=8)
    # 标注泡棉位置
    for gap in [0.5, 1.5, 2.5, 4.5, 5.5, 6.5]:
        ax3.axvline(gap, color="orange", lw=0.8, ls="--", alpha=0.5)
    ax3.text(3.5, ax3.get_ylim()[1]*0.9, "泡沫间隙 →", color="orange", fontsize=7, ha="center")
    fig2.tight_layout()
    png2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "ecm_pack_8s_foam_bar.png")
    png2 = os.path.abspath(png2)
    fig2.savefig(png2, dpi=130)
    plt.close(fig2)
    print(f"  温升柱状图: {png2}")

    # ────── 图 3: 最热电芯的 3D 温度场 ──────
    hottest_idx = int(np.argmax(T_all_end))
    T_hot = T_all_end[hottest_idx]
    # 估算最热芯的产热: 同串各芯 I 相同, 取它自己的电流
    Q_avg = float(np.mean(np.abs(I_cell[-30:, hottest_idx])**2 * 0.4e-3))
    print(f"\n  最热电芯: bat{hottest_idx+1} (T={T_hot:.2f}K), 估算产热≈{Q_avg:.1f}W")

    # 创建单芯 3D 热模型，用 pack 结果约束边界条件
    tm = CellThermalModel(
        Lx=0.174, Ly=0.0717, Lz=0.207, dim=3,
        nx=6, ny=5, nz=8,
        rho=2300.0, cp=1000.0, k=(12.0, 0.7, 11.6),
        h={"default": 5.0},        # 假设暴露面自然对流
        T_amb=298.15, T_init=298.15,
        R_shell=0.3,
    )
    # 用 pack 平均产热驱动 3D 回放
    for _ in range(int(t_total / dt)):
        tm.step(Q_avg, dt)
    stats = tm.temperature_stats()
    print(f"  3D热场: T_avg={stats['T_avg [K]']:.2f}K  "
          f"T_max={stats['T_max [K]']:.2f}K  "
          f"ΔT={stats['dT_max [K]']:.4f}K  "
          f"T_surface={stats.get('T_surface [K]', 'N/A')}")

    fig3 = tm.plot_summary(
        save_path="/workspace/ecm_pack_8s_foam_3dcell.png", dpi=130)
    print(f"  3D电芯温度场: /workspace/ecm_pack_8s_foam_3dcell.png")

    # ────── CSV 输出 ──────
    csv_path = "/workspace/ecm_pack_8s_foam_data.csv"
    header = "Time_s,Vt_V," + ",".join([f"T_bat{i+1}_K" for i in range(n_cells)]) \
             + "," + ",".join([f"SOC_bat{i+1}" for i in range(n_cells)])
    np.savetxt(csv_path,
               np.column_stack([t, Vt] + [T_cells[:, i] for i in range(n_cells)]
                               + [soc[:, i] for i in range(n_cells)]),
               delimiter=",", header=header, fmt="%.6f")
    print(f"  数据: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

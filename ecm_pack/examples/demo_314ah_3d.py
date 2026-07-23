# demo_314ah_3d.py  ——  314Ah 大电芯 + 三维热模型演示
#
# 演示：
#   1) 使用内置 314Ah 储能大电芯默认参数（容量、OCV、R0、几何、热物性）；
#   2) 电芯 ECM 电气模型与三维热模型耦合：ECM 算产热 -> 三维 FVM 求解温度场；
#   3) 输出电芯内部温度分布（最高温、最大温差、热点位置）。
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


def main():
    # ---- 1) 314Ah 大电芯默认规格 ----
    spec = ep.cell_314ah_spec(soc_init=0.5, T_init=298.15)
    cell = ep.ECMCell(spec)
    print("===== 314Ah 大电芯 + 三维热模型 =====")
    print(f"容量: {spec.capacity} Ah")
    print(f"尺寸: {spec.Lx*1e3:.0f}×{spec.Ly*1e3:.0f}×{spec.Lz*1e3:.0f} mm  "
          f"(体积 {spec.Lx*spec.Ly*spec.Lz*1e3:.2f} L)")
    print(f"密度: {spec.rho} kg/m³, 比热: {spec.cp} J/(kg·K)")
    print(f"导热: k={spec.k} W/(m·K) (各向异性)")
    print(f"初始 SoC: {cell.soc}, OCV: {cell.ocv():.3f} V")

    # ---- 2) 三维热模型 ----
    th3d = ep.Cell3DThermal(
        Lx=spec.Lx, Ly=spec.Ly, Lz=spec.Lz,
        nx=5, ny=5, nz=8,  # 200 个网格点
        rho=spec.rho, cp=spec.cp, k=spec.k,
        h=8.0, T_amb=298.15, T_init=298.15,
    )
    print(f"\n三维网格: {th3d.nx}×{th3d.ny}×{th3d.nz} = {th3d.N} 节点")
    print(f"网格尺寸: dx={th3d.dx*1e3:.1f} mm, dy={th3d.dy*1e3:.1f} mm, dz={th3d.dz*1e3:.1f} mm")

    # ---- 3) 耦合仿真：1C 放电 600s（10 min）----
    I_1c = 314.0  # 1C 电流 = 314A
    dt = 2.0
    n_steps = 300  # 600s = 10min

    t_hist, T_max_hist, T_avg_hist, V_hist, soc_hist = [], [], [], [], []
    for s in range(n_steps):
        R0 = cell.step_electrical(I_1c, dt)
        Q = cell.heat(I_1c, R0)
        # 体积产热率
        q_vol = Q / th3d.volume
        th3d.step(Q, dt, t=(s+1)*dt)
        cell.T = th3d.T_avg  # 反馈平均温度给 ECM（影响 R0/OCV）
        if (s+1) % 10 == 0:
            t_hist.append((s+1)*dt)
            T_max_hist.append(th3d.T_max)
            T_avg_hist.append(th3d.T_avg)
            V_hist.append(cell.terminal_voltage(R0, I_1c))
            soc_hist.append(cell.soc)

    stats = th3d.temperature_stats()
    print(f"\n--- 1C 放电 {n_steps*dt}s 后 ---")
    print(f"SoC: {cell.soc:.4f}, 端电压: {cell.terminal_voltage(R0, I_1c):.3f} V")
    print(f"总产热: {Q:.1f} W, 体积产热率: {q_vol/1e3:.2f} kW/m³")
    print(f"温度场: 最高 {stats['T_max [K]']:.2f} K, 最低 {stats['T_min [K]']:.2f} K, "
          f"平均 {stats['T_avg [K]']:.2f} K, 最大温差 {stats['dT_max [K]']:.2f} K")

    # ---- 4) 画图 ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    axs[0].plot(np.array(t_hist)/60, T_max_hist, "r-", label="T_max")
    axs[0].plot(np.array(t_hist)/60, T_avg_hist, "b-", label="T_avg")
    axs[0].axhline(298.15, color="gray", ls="--", label="T_amb")
    axs[0].set_title("电芯温度演化 (1C 放电)")
    axs[0].set_xlabel("t [min]"); axs[0].set_ylabel("T [K]"); axs[0].legend()

    # 三维温度场中心切片（x 中截面）
    T3d = th3d.reshape()
    im = axs[1].imshow(
        T3d[th3d.nx//2, :, :].T,  # y-z 平面
        aspect="auto", origin="lower",
        extent=[0, th3d.Ly*1e3, 0, th3d.Lz*1e3],
        cmap="hot",
    )
    axs[1].set_title(f"温度场切片 (x={th3d.Lx*1e3/2:.0f}mm 中截面)")
    axs[1].set_xlabel("y [mm]"); axs[1].set_ylabel("z [mm]")
    plt.colorbar(im, ax=axs[1], label="T [K]")

    axs[2].plot(np.array(t_hist)/60, V_hist, "g-")
    axs[2].set_title("端电压演化")
    axs[2].set_xlabel("t [min]"); axs[2].set_ylabel("V [V]")

    fig.suptitle(f"314Ah 储能大电芯 三维热模型 (1C 放电 10min, ΔT_max={stats['dT_max [K]']:.1f}K)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("/workspace/ecm_pack_314ah_3d_demo.png", dpi=130)
    print(f"\n图表已保存 -> /workspace/ecm_pack_314ah_3d_demo.png")


if __name__ == "__main__":
    main()

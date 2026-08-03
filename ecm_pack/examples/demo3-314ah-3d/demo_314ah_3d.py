# demo_314ah_3d.py  —�?  314Ah 大电�? + 多维热模�?(1D/2D/3D)演示

# 演示�?
# 1) 使用内置 314Ah 储能大电芯默认参数（用户指定的精确参数）�?
# 2) 同一电芯分别�? 1D / 2D / 3D 热模型求解，对比温度场差异；
# 3) 展示三维温度场内部分布切片�?
import os
import sys
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import ecm_pack as ep
def run_one(dim, nx, ny, nz, I_load=314.0, dt=2.0, n_steps=300):
    """用指定维度热模型跑一次耦合仿真，返回时序与末态统计�?"""
    spec = ep.cell_314ah_spec(soc_init=0.5, T_init=298.15)
    cell = ep.ECMCell(spec)
    th = ep.CellThermalModel(
        Lx=spec.Lx, Ly=spec.Ly, Lz=spec.Lz,
        dim=dim, nx=nx, ny=ny, nz=nz,
        rho=spec.rho, cp=spec.cp, k=spec.k,
        h=8.0, T_amb=298.15, T_init=298.15,
    )
    t_hist, T_max_h, T_avg_h, V_h = [], [], [], []
    for s in range(n_steps):
        R0 = cell.step_electrical(I_load, dt)
        Q = cell.heat(I_load, R0)
        th.step(Q, dt, t=(s + 1) * dt)
        cell.T = th.T_avg
        if (s + 1) % 10 == 0:
            t_hist.append((s + 1) * dt)
            T_max_h.append(th.T_max)
            T_avg_h.append(th.T_avg)
            V_h.append(cell.terminal_voltage(R0, I_load))
    return {
        "th": th, "cell": cell,
        "t": np.array(t_hist), "T_max": np.array(T_max_h),
        "T_avg": np.array(T_avg_h), "V": np.array(V_h),
    }
def main():
    print("===== 314Ah 大电�? + 多维热模�?(1D/2D/3D) =====")
    spec = ep.cell_314ah_spec(soc_init=0.5)
    print(f"容量: {spec.capacity} Ah")
    print(f"尺寸: 宽{spec.Lx*1e3:.1f} × 厚{spec.Ly*1e3:.1f} × 高{spec.Lz*1e3:.1f} mm  "*
    f"(体积 {spec.Lxspec.Ly*spec.Lz*1e3:.2f} L)")
    print(f"密度: {spec.rho} kg/m³, 比热: {spec.cp} J/(kg·K)")
    print(f"导热: kx={spec.k[0]} ky={spec.k[1]} kz={spec.k[2]} W/(m·K) (各向异�?)")
    print("内阻: R0=0.4mΩ, R1=0.4mΩ, τ=100s")
    print(f"初始 SoC: {spec.soc_init}, OCV: {ep.ECMCell(spec).ocv():.3f} V")
    print()
    # 三种维度对比
    configs = [
        ("1D (仅X宽度方向)", 1, 20, 1, 1),
        ("2D (XY平面)",      2, 12, 8, 1),
        ("3D (完整三维)",    3, 6, 5, 8),
    ]
    results = {}
    for label, dim, nx, ny, nz in configs:
        r = run_one(dim, nx, ny, nz)
        results[label] = r
        stats = r["th"].temperature_stats()
        print(f"--- {label} (网格 {r['th'].nx}×{r['th'].ny}×{r['th'].nz}={r['th'].N}) ---")
        print(f"  末�?: T_max={stats['T_max [K]']:.3f} T_avg={stats['T_avg [K]']:.3f} "
              f"dT={stats['dT_max [K]']:.3f}K  V={r['V'][-1]:.3f}V  SoC={r['cell'].soc:.4f}")

    # ---- 画图 ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"1D (仅X宽度方向)": "tab:blue", "2D (XY平面)": "tab:orange", "3D (完整三维)": "tab:red"}

    for label, r in results.items():
        t_min = r["t"] / 60
        axs[0].plot(t_min, r["T_max"], color=colors[label], label=f"{label} T_max")
        axs[0].plot(t_min, r["T_avg"], color=colors[label], ls="--", alpha=0.7)
    axs[0].axhline(298.15, color="gray", ls=":", label="T_amb")
    axs[0].set_title("温度演化对比 (1C 放电)")
    axs[0].set_xlabel("t [min]"); axs[0].set_ylabel("T [K]"); axs[0].legend(fontsize=8)

    # 3D 温度场切�?
    th3d = results["3D (完整三维)"]["th"]
    T3d = th3d.reshape()
    im = axs[1].imshow(
        T3d[:, th3d.ny // 2, :].T,  # X-Z 平面（厚度中截面�?
        aspect="auto", origin="lower",
        extent=[0, th3d.Lx * 1e3, 0, th3d.Lz * 1e3],
        cmap="hot",
    )
    axs[1].set_title(f"3D温度场切�? (Y={th3d.Ly*1e3/2:.1f}mm)")
    axs[1].set_xlabel("X 宽度 [mm]"); axs[1].set_ylabel("Z 高度 [mm]")
    plt.colorbar(im, ax=axs[1], label="T [K]")

    # 端电压对�?
    for label, r in results.items():
        axs[2].plot(r["t"] / 60, r["V"], color=colors[label], label=label)
    axs[2].set_title("端电压演�?")
    axs[2].set_xlabel("t [min]"); axs[2].set_ylabel("V [V]"); axs[2].legend(fontsize=8)

    fig.suptitle("314Ah 大电芯多维热模型对比 (1D/2D/3D, 1C 放电 10min)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "ecm_pack_314ah_3d_demo.png"), dpi=130)

if __name__ == "__main__":
    main()

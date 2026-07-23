# topology_demo.py  ——  拓扑热切换演示
#
# 用户场景：一组 8S 先工作 10 min，之后另一组 8S 并入，两组并联共同工作。
# 这验证 ecm_pack 的「可重构电池包」能力：
#   1) 电芯状态(SoC/RC/温度)与网表解耦 —— 切换只换网表与 active 映射；
#   2) 被断开的 group B 在待机期间状态完全冻结；
#   3) group B 并入瞬间，MNA 自动产生组间环流/浪涌（高 SoC 组给低 SoC 组充电）；
#   4) 之后两组 SoC 被均衡，整包容量/电流能力翻倍。
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 注册系统中文字体（Noto Sans CJK），避免图里中文显示为方块
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ecm_pack as ep


# ----------------------------------------------------------------------------
# 复用 demo 的可定制 ECM 规格（工业风格）
# ----------------------------------------------------------------------------
def make_ocv():
    s = np.linspace(0, 1, 101)
    v = 3.2 + 0.95 * s + 0.08 * np.sin(2 * np.pi * s) + 0.1 * s**2
    return ep.lookup_1d(s, v)


def R0_of(Tdeg, I, soc):
    return 0.01 * (1.0 + 0.6 * (1.0 - soc)) * np.exp(2000.0 * (1.0 / 298.15 - 1.0 / (Tdeg + 273.15))) + 1e-5 * abs(I)


def R1_of(Tdeg, I, soc):
    return 0.5 * R0_of(Tdeg, I, soc)


def C1_of(Tdeg, I, soc):
    return 6000.0


def dUdT_of(Tdeg, soc):
    return -1e-4 + 2e-4 * soc


def build_spec(soc_init=1.0, T_init=298.15):
    return ep.ECMCellSpec(
        capacity=5.0,
        ocv=make_ocv(), R0=R0_of, R=[R1_of], C=[C1_of], dUdT=dUdT_of,
        soc_init=soc_init, T_init=T_init,
    )


def main():
    nS = 8
    # group A: cells 0..7 先工作；group B: cells 8..15 待机后并入
    specs = [build_spec(soc_init=0.80, T_init=298.15) for _ in range(nS)] + \
            [build_spec(soc_init=1.00, T_init=298.15) for _ in range(nS)]
    cells = [ep.ECMCell(sp) for sp in specs]

    nl_solo, act_solo, nl_par, act_par = ep.setup_two_group(nS)

    # 自定义电芯间导热：16 芯沿一条链相邻导热 G=5 W/K（演示需求 4 仍可用）
    conduction = [(i, i + 1, 5.0) for i in range(2 * nS - 1)]
    thermal = ep.ThermalNetwork(2 * nS, C_th=800.0, h=0.5, T_amb=298.15, conduction=conduction)

    pack = ep.Pack(cells, nl_solo, thermal=thermal, v_cut_lower=2.5, active=act_solo)

    # 整包恒定电流放电：solo 时 8S 独扛，并入后两组分摊
    I_pack = 8.0  # A
    dt = 1.0
    t_switch = 600.0  # 10 min
    n_steps = 1200    # 总时长 20 min

    out = pack.solve(
        dt=dt, control=I_pack, control_type="current", n_steps=n_steps,
        record_every=5,
        topology_events=[(t_switch, nl_par, act_par)],  # 定时切换
    )

    topo_changes = out["Topology changes [s]"]
    print("\n===== 拓扑热切换演示：8S 先工作 10min，另一组 8S 随后并入并联 =====")
    print(f"拓扑切换时刻(s): {topo_changes}")
    print(f"是否提前截止: {out['Terminated']} (step={out['Term step']})")

    # 端口电压：切换前后
    t = out["Time [s]"]
    Vt = out["Pack terminal voltage [V]"]
    i_pre = np.argmin(np.abs(t - (t_switch - dt)))   # 切换前最后一步
    i_post = np.argmin(np.abs(t - (t_switch + dt)))  # 切换后第一步
    print(f"\n端口电压: 切换前 {Vt[i_pre]:.3f} V -> 切换后 {Vt[i_post]:.3f} V")

    # SoC 均衡：A 组(cell0) 与 B 组(cell8)
    socA0 = out["Cell SoC"][:, 0]
    socB0 = out["Cell SoC"][:, 8]
    i_switch_rec = int(t_switch / dt / 5)  # record_every=5
    print(f"\n切换瞬间 SoC: A0={socA0[i_switch_rec]:.4f}, B0={socB0[i_switch_rec]:.4f} "
          f"(Δ={socB0[i_switch_rec]-socA0[i_switch_rec]:.4f})")
    print(f"末态 SoC:     A0={socA0[-1]:.4f}, B0={socB0[-1]:.4f} "
          f"(Δ={socB0[-1]-socA0[-1]:.4f})  <- Δ 明显缩小即完成均衡")

    # 浪涌/环流：切换后第一步 cell 电流
    IA = out["Cell current [A]"]
    print(f"\n切换后 cell 电流(浪涌): A0={IA[i_switch_rec,0]:.2f} A, B0={IA[i_switch_rec,8]:.2f} A, "
          f"pack={out['Pack current [A]'][i_switch_rec]:.2f} A")
    print(f"末态 cell 电流:         A0={IA[-1,0]:.2f} A, B0={IA[-1,8]:.2f} A")

    # 温度（自定义导热下被拉平）
    T = out["Cell temperature [K]"]
    print(f"\n末态温度 K 范围: [{T[-1].min():.2f}, {T[-1].max():.2f}]  (相邻导热拉平)")

    # ------------------------------------------------------------------
    # 画图
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(t / 60.0, Vt, color="tab:blue")
    axs[0, 0].axvline(t_switch / 60.0, color="red", ls="--", lw=1, label="topology switch")
    axs[0, 0].set_title("Pack terminal voltage")
    axs[0, 0].set_xlabel("t [min]"); axs[0, 0].set_ylabel("V"); axs[0, 0].legend()

    axs[0, 1].plot(t / 60.0, socA0, label="cell0 (group A)")
    axs[0, 1].plot(t / 60.0, socB0, label="cell8 (group B)")
    axs[0, 1].axvline(t_switch / 60.0, color="red", ls="--", lw=1)
    axs[0, 1].set_title("SoC: A0 vs B0 (均衡收敛)")
    axs[0, 1].set_xlabel("t [min]"); axs[0, 1].set_ylabel("SoC"); axs[0, 1].legend()

    axs[1, 0].plot(t / 60.0, IA[:, 0], label="cell0 (A)")
    axs[1, 0].plot(t / 60.0, IA[:, 8], label="cell8 (B)")
    axs[1, 0].axvline(t_switch / 60.0, color="red", ls="--", lw=1)
    axs[1, 0].set_title("Cell current: 并入瞬间出现浪涌/环流")
    axs[1, 0].set_xlabel("t [min]"); axs[1, 0].set_ylabel("I [A]"); axs[1, 0].legend()

    axs[1, 1].plot(t / 60.0, T[:, 0], label="cell0")
    axs[1, 1].plot(t / 60.0, T[:, 8], label="cell8")
    axs[1, 1].axvline(t_switch / 60.0, color="red", ls="--", lw=1)
    axs[1, 1].set_title("Cell temperature (自定义导热)")
    axs[1, 1].set_xlabel("t [min]"); axs[1, 1].set_ylabel("T [K]"); axs[1, 1].legend()

    fig.suptitle("ecm_pack — 拓扑热切换: 8S 工作 10min 后另一组 8S 并入并联", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("/workspace/ecm_pack_topology_demo.png", dpi=130)
    print("\n图表已保存 -> /workspace/ecm_pack_topology_demo.png")


if __name__ == "__main__":
    main()

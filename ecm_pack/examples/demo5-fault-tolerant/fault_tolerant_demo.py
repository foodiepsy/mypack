# fault_tolerant_demo.py  ——  故障容错拓扑重构演示（事件驱动切换）
#
# 用户强调的能力："随时进行拓扑切换"。最典型的应用是**故障容错 BMS**：
# 某只电芯快到截止电压时，控制器自动把它旁路（短接移除），整包以 (n-1)
# 串联继续工作，而不是整体停机。
#
# 本演示：8 芯串联 (8S) 放电。cell3 是缺陷电芯（容量只有 2Ah，远小于其他
# 5Ah），会先逼近放电截止。当其端电压跌到 3.4V（尚远高于硬性安全下限 2.0V）
# 时，switch_callback 自动把整包重构为「旁路 cell3 的 7S」，放电继续进行。
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


def make_ocv():
    s = np.linspace(0, 1, 101)
    v = 3.2 + 0.95 * s + 0.08 * np.sin(2 * np.pi * s) + 0.1 * s**2
    return ep.lookup_1d(s, v)


def R0_of(Tdeg, I, soc):
    return 0.01 * (1 + 0.6 * (1 - soc)) * np.exp(2000 * (1 / 298.15 - 1 / (Tdeg + 273.15))) + 1e-5 * abs(I)


def R1_of(Tdeg, I, soc):
    return 0.5 * R0_of(Tdeg, I, soc)


def C1_of(Tdeg, I, soc):
    return 6000.0


def dUdT_of(Tdeg, soc):
    return -1e-4 + 2e-4 * soc


def build_spec(soc_init=1.0, capacity=5.0, T_init=298.15):
    return ep.ECMCellSpec(
        capacity=capacity,
        ocv=make_ocv(), R0=R0_of, R=[R1_of], C=[C1_of], dUdT=dUdT_of,
        soc_init=soc_init, T_init=T_init,
    )


def main():
    n = 8
    FAULT = 3  # 缺陷电芯下标
    # 正常 5Ah，cell3 仅 2Ah（放电更快，先到截止）
    specs = [build_spec(soc_init=0.9, capacity=5.0) for _ in range(n)]
    specs[FAULT] = build_spec(soc_init=0.9, capacity=2.0)
    cells = [ep.ECMCell(sp) for sp in specs]

    nl0, act0 = ep.setup_series_bypass(n)  # 初始 8S
    pack = ep.Pack(cells, nl0, active=act0, v_cut_lower=2.0)  # 硬性安全下限设低，由回调决定旁路时机

    # 事件驱动切换：缺陷芯端电压跌到阈值即自动旁路
    I_load = 10.0
    state = {"bypassed": False, "t_bypass": None}

    def cb(t, p):
        if state["bypassed"]:
            return None
        c = p.cells[FAULT]
        R0 = c.spec.R0(c.T - 273.15, I_load, c.soc)
        Vterm = c.terminal_voltage(R0, I_load)
        if Vterm < 3.4:  # 逼近截止但尚未触及硬性下限
            nl, act = ep.setup_series_bypass(n, bypass_idx=FAULT)
            state["bypassed"] = True
            state["t_bypass"] = t
            return nl, act
        return None

    out = pack.solve(
        dt=1.0, control=I_load, control_type="current", n_steps=800,
        record_every=4, switch_callback=cb,
    )

    t = out["Time [s]"]
    Vt = out["Pack terminal voltage [V]"]
    t_bp = state["t_bypass"]
    print("\n===== 故障容错重构：缺陷芯(cell%d)自动旁路，整包继续放电 =====" % FAULT)
    print(f"旁路发生时刻: t={t_bp:.0f}s  (缺陷芯端电压阈值 3.4V)")
    print(f"是否提前终止: {out['Terminated']}  (step={out['Term step']})")
    if t_bp is not None:
        i_b = int(np.argmin(np.abs(t - t_bp)))
        print(f"端口电压: 旁路前 {Vt[max(0,i_b-1)]:.2f} V -> 旁路后 {Vt[min(len(Vt)-1,i_b+1)]:.2f} V "
              f"(少一芯, 电压自然下降)")
    print(f"末态各芯 SoC: {np.round(out['Cell SoC'][-1], 3)}")
    print(f"  cell{FAULT} 已旁路 -> SoC 冻结在 {out['Cell SoC'][-1, FAULT]:.3f}, 电流记 0")

    # ------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(t / 60.0, Vt, color="tab:blue")
    if t_bp is not None:
        axs[0, 0].axvline(t_bp / 60.0, color="red", ls="--", lw=1, label="bypass cell%d" % FAULT)
    axs[0, 0].set_title("Pack terminal voltage (旁路后电压下降但持续放电)")
    axs[0, 0].set_xlabel("t [min]"); axs[0, 0].set_ylabel("V"); axs[0, 0].legend()

    axs[0, 1].plot(t / 60.0, out["Cell SoC"], label=[f"c{i}" for i in range(n)])
    if t_bp is not None:
        axs[0, 1].axvline(t_bp / 60.0, color="red", ls="--", lw=1)
    axs[0, 1].set_title("各电芯 SoC (缺陷芯旁路后冻结)")
    axs[0, 1].set_xlabel("t [min]"); axs[0, 1].set_ylabel("SoC"); axs[0, 1].legend()

    axs[1, 0].plot(t / 60.0, out["Cell terminal voltage [V]"], label=[f"c{i}" for i in range(n)])
    if t_bp is not None:
        axs[1, 0].axvline(t_bp / 60.0, color="red", ls="--", lw=1)
    axs[1, 0].set_title("各电芯端电压 (cell%d 先跌, 旁路后移除)".format(FAULT))
    axs[1, 0].set_xlabel("t [min]"); axs[1, 0].set_ylabel("V"); axs[1, 0].legend()

    axs[1, 1].plot(t / 60.0, out["Cell current [A]"], label=[f"c{i}" for i in range(n)])
    if t_bp is not None:
        axs[1, 1].axvline(t_bp / 60.0, color="red", ls="--", lw=1)
    axs[1, 1].set_title("各电芯电流 (旁路后 cell%d 电流=0)".format(FAULT))
    axs[1, 1].set_xlabel("t [min]"); axs[1, 1].set_ylabel("I [A]"); axs[1, 1].legend()

    fig.suptitle("ecm_pack — 故障容错重构: 缺陷芯端电压到阈值即自动旁路, 整包继续工作", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "ecm_pack_fault_tolerant_demo.png"), dpi=130)
    print("\n图表已保存 -> /workspace/ecm_pack_fault_tolerant_demo.png")


if __name__ == "__main__":
    main()

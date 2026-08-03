# demo.py  —�?  ecm_pack 功能演示
# 覆盖�?1) ECM 模型定制  2) 自定义串并联拓扑  3) 热模型集�?  4) 自定义电芯间导热
import os
import sys
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
# 注册系统中文字体，避免图里中文显示为方块
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    fm.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import ecm_pack as ep
# ----------------------------------------------------------------------------
# 0) 定义一个「工业风格」的可定�? ECM 电芯规格
# OCV 用曲线；R0/R1 �? (温度, 电流, SoC) 变化；含熵热 dU/dT�?1 �? RC�?
# ----------------------------------------------------------------------------
def make_ocv():
    s = np.linspace(0, 1, 101)
    v = 3.2 + 0.95 * s + 0.08 * np.sin(2 * np.pi * s) + 0.1 * s**2
    return ep.lookup_1d(s, v)
def R0_of(Tdeg, I, soc):
    base = 0.01
    soc_term = 1.0 + 0.6 * (1.0 - soc)
    arr = np.exp(2000.0 * (1.0 / 298.15 - 1.0 / (Tdeg + 273.15)))
    return base * soc_term * arr + 1e-5 * abs(I)
def R1_of(Tdeg, I, soc):
    return 0.5 * R0_of(Tdeg, I, soc)
def C1_of(Tdeg, I, soc):
    return 6000.0
def dUdT_of(Tdeg, soc):
    return -1e-4 + 2e-4 * soc
def build_spec(soc_init=1.0, T_init=298.15):
    return ep.ECMCellSpec(
        capacity=5.0,
        ocv=make_ocv(),
        R0=R0_of,
        R=[R1_of],
        C=[C1_of],
        dUdT=dUdT_of,
        soc_init=soc_init,
        T_init=T_init,
    )
# ----------------------------------------------------------------------------
# 1) ECM 模型定制：单芯验证（放电 SoC 下降、电压跌落、产热升温）
# ----------------------------------------------------------------------------
def demo_single_cell():
    spec = build_spec(soc_init=0.8)
    cell = ep.ECMCell(spec)
    dt, I = 10.0, 5.0
    soc0, v0 = cell.soc, cell.voltage_behind_R0()
    T0 = cell.T
    for _ in range(360):  # 1h @ 5A
        R0 = cell.step_electrical(I, dt)
        cell.T += cell.heat(I, R0) / 800.0 * dt  # 单芯自行加一点热
    print("\n[单芯 ECM] 5A 放电 1h:")
    print(f"  SoC: {soc0:.3f} -> {cell.soc:.3f}  "
          f"电压: {v0:.3f} -> {cell.terminal_voltage(R0, I):.3f} V")
    print(f"  温度: {T0:.2f} -> {cell.T:.2f} K  (温升 {cell.T - T0:+.2f} K)")
# ----------------------------------------------------------------------------
# 2) 自定义串并联拓扑�?2s2p，四芯初�? SoC 故意不均�? -> 验证并联不均�?
# ----------------------------------------------------------------------------
def demo_2s2p():
    specs = [build_spec(soc_init=s) for s in (1.00, 0.90, 0.80, 0.70)]
    cells = [ep.ECMCell(sp) for sp in specs]
    netlist, _, _ = ep.setup_circuit(2, 2)  # 2 �? 2 �?
    thermal = ep.ThermalNetwork(4, C_th=800.0, h=5.0, T_amb=298.15)
    pack = ep.Pack(cells, netlist, thermal=thermal, v_cut_lower=2.5)
    # 整包 4 A 放电，dt=10s�?180 �? = 30 min（温和，避免过早截止�?
    out = pack.solve(dt=10.0, control=4.0, control_type="current", n_steps=180, record_every=10)
    print("\n[2s2p 不均衡放电] 整包 4A:")
    print(f"  端口电压: {out['Pack terminal voltage [V]'][0]:.3f} -> "
          f"{out['Pack terminal voltage [V]'][-1]:.3f} V")
    print(f"  各芯 SoC 末�?: {np.round(out['Cell SoC'][-1], 3)}")
    I = out["Cell current [A]"]
    mid = min(9, I.shape[0] - 2)  # 取中途一步展示并联不均流
    print(f"  并联不均�?(中途步 各串级内�?): "
          f"{np.round(np.abs(I[mid,1]-I[mid,0]),3)} / {np.round(np.abs(I[mid,3]-I[mid,2]),3)} A")
    return out
# ----------------------------------------------------------------------------
# 3) + 4) 热模型集�? �? 自定义电芯间导热
# �? 1s2p：两芯初�? SoC 不同 -> 产热不同�?
# 场景 A：彼此绝热（仅对环境对流�?-> 两芯温度保持差异
# 场景 B：两芯强耦合（自定义导热�?-> 温度被拉�?
# ----------------------------------------------------------------------------
def demo_thermal():
    def run(conduction, label):
        specs = [build_spec(soc_init=0.95, T_init=298.15),
                 build_spec(soc_init=0.60, T_init=298.15)]
        cells = [ep.ECMCell(sp) for sp in specs]
        netlist, _, _ = ep.setup_circuit(1, 2)  # 1 �? 2 �?
        thermal = ep.ThermalNetwork(2, C_th=300.0, h=0.5, T_amb=298.15, conduction=conduction)
        pack = ep.Pack(cells, netlist, thermal=thermal, v_cut_lower=2.5)
        out = pack.solve(dt=10.0, control=6.0, control_type="current", n_steps=360, record_every=10)
        print(f"\n[{label}] 末态温�? K: {np.round(out['Cell temperature [K]'][-1], 3)}")
        return out
    out_adiabatic = run(None, "�?-绝热(仅对�?)")
    out_coupled = run([(0, 1, 30.0)], "�?-自定义导�?(0-1 强耦合 G=30)")
    return out_adiabatic, out_coupled

# ----------------------------------------------------------------------------
# 5) 功率控制模式（不同控制方式）
# ----------------------------------------------------------------------------
def demo_power():
    specs = [build_spec(soc_init=0.9) for _ in range(4)]
    cells = [ep.ECMCell(sp) for sp in specs]
    netlist, _, _ = ep.setup_circuit(2, 2)
    pack = ep.Pack(cells, netlist, v_cut_lower=2.5)
    out = pack.solve(dt=10.0, control=120.0, control_type="power", n_steps=360, record_every=10)
    k = -1
    print("\n[功率控制 120W] 末�?: "
          f"V={out['Pack terminal voltage [V]'][k]:.2f}V "
          f"I={out['Pack current [A]'][k]:.2f}A "
          f"P={out['Pack power [W]'][k]:.1f}W")
    return out
# ----------------------------------------------------------------------------
# 画图汇�?
# ----------------------------------------------------------------------------
def plot_all(out_2s2p, out_adiabatic, out_coupled):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    t = out_2s2p["Time [s]"] / 60.0
    axs[0, 0].plot(t, out_2s2p["Pack terminal voltage [V]"], label="Pack V")
    axs[0, 0].set_title("2s2p 端口电压"); axs[0, 0].set_xlabel("t [min]"); axs[0, 0].set_ylabel("V"); axs[0, 0].legend()
    axs[0, 1].plot(t, out_2s2p["Cell SoC"], label=[f"cell{i}" for i in range(4)])
    axs[0, 1].set_title("各电�? SoC（不均衡放电�?"); axs[0, 1].set_xlabel("t [min]"); axs[0, 1].set_ylabel("SoC"); axs[0, 1].legend()

    tc = out_coupled["Cell temperature [K]"]
    ta = out_adiabatic["Cell temperature [K]"]
    tc = tc[: min(len(tc), len(ta))]
    t2 = out_adiabatic["Time [s]"][: len(tc)] / 60.0
    axs[1, 0].plot(t2, ta[:len(tc), 0], label="adiabatic cell0")
    axs[1, 0].plot(t2, ta[:len(tc), 1], label="adiabatic cell1")
    axs[1, 0].plot(t2, tc[:, 0], "--", label="coupled cell0")
    axs[1, 0].plot(t2, tc[:, 1], "--", label="coupled cell1")
    axs[1, 0].set_title("温度：绝�? vs 自定义导热（不同产热的电芯被拉平�?")
    axs[1, 0].set_xlabel("t [min]"); axs[1, 0].set_ylabel("T [K]"); axs[1, 0].legend()

    axs[1, 1].plot(t, out_2s2p["Cell current [A]"], label=[f"cell{i}" for i in range(4)])
    axs[1, 1].set_title("各电芯电流（并联不均流）"); axs[1, 1].set_xlabel("t [min]"); axs[1, 1].set_ylabel("I [A]"); axs[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "ecm_pack_demo.png"), dpi=130)

if __name__ == "__main__":
    demo_single_cell()
    out_2s2p = demo_2s2p()
    out_adiabatic, out_coupled = demo_thermal()
    demo_power()
    plot_all(out_2s2p, out_adiabatic, out_coupled)
    print("\n全部演示完成�?")

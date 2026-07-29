# demo_8s2p_200a.py  ——  8S2P / 总负载 200A 工况：重点考察「环流」
#
# 用户场景：
#   一组 8S（支路 A，电芯 0..7）先单独承受 200A 总负载工作 0.1 h（=360 s），
#   期间支路 B（电芯 8..15）闲时待机、保持 SoC=1.0 不变；
#   随后 t=360 s 支路 B 从闲时并入，整包变为 8S2P，两组并联共同承担 200A 总负载。
#
# 关注点：环流 (circulation)
#   并入瞬间，A 组已放电到 SoC≈0.936（OCV 偏低），B 组仍为 1.0（OCV 偏高）。
#   两组在电气上并联，MNA 自动解算出一个「组间环流」：
#     - 高 SoC 的 B 组向低 SoC 的 A 组倒灌电流（给 A 充电/减缓 A 放电）；
#     - 环流大小 = (I_B - I_A) / 2，随时间按两组的 SoC（电压）差指数衰减直至均衡。
#
# 使用用户指定的 314Ah 大电芯默认参数（cell_314ah_spec）：
#   R0=R1=0.4 mΩ, τ=100 s, C1=250 kF；尺寸 174×71.7×207 mm；
#   rho=2300, cp=1000, k=(12, 0.7, 11.6) W/(m·K)。
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import ecm_pack as ep


def main():
    # ---- 工况参数 ----
    nS = 8                       # 每支路串数（支路 A = 0..7，支路 B = 8..15）
    N = 2 * nS                   # 16 颗电芯
    I_load = 200.0               # 总负载电流 [A]
    t_solo = 0.1 * 3600.0        # 支路 A 单独工作时间 = 0.1 h = 360 s
    dt = 1.0                     # 步长 [s]
    t_post = 600.0               # 并入后再跑 600 s，观察环流衰减与 SoC 均衡
    n_steps = int(round(t_solo + t_post))
    record_every = 1

    print("=" * 70)
    print(" 8S2P / 总负载 200A 工况 —— 重点考察「环流」")
    print("=" * 70)
    print(f" 电芯规格         : 314Ah 大电芯（cell_314ah_spec）")
    print(f" 支路 A (电芯0..7): 单独承担 200A 工作 {t_solo:.0f}s ({t_solo/3600:.2f}h)")
    print(f" 支路 B (电芯8..15): 闲时待机，SoC 保持 1.0")
    print(f" 切换时刻 t={t_solo:.0f}s : 支路 B 并入 → 8S2P，两组并联共担 200A")
    print(f" 步长 dt={dt}s, 总步数={n_steps} (含并入后 {t_post:.0f}s)")
    print("=" * 70)

    # ---- 构建 16 颗电芯，全部初始 SoC=1.0 ----
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15)) for _ in range(N)]

    # ---- 拓扑：单组 8S + 两组 8S 并联(8S2P) ----
    nl_solo, act_solo, nl_par, act_par = ep.setup_two_group(nS)
    assert act_par == [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15], act_par

    # ---- Pack：先以单组 8S 工作，定时在 t_solo 切换为 8S2P ----
    pack = ep.Pack(cells, nl_solo, active=act_solo, v_cut_lower=2.0, v_cut_upper=4.4)

    out = pack.solve(
        dt=dt,
        control=I_load,                 # 全程总负载 200A（电流控制）
        control_type="current",
        n_steps=n_steps,
        record_every=record_every,
        topology_events=[(t_solo, nl_par, act_par)],
    )

    t = out["Time [s]"]
    I_cell = out["Cell current [A]"]    # shape (n_rec, N)
    soc = out["Cell SoC"]               # shape (n_rec, N)
    Vt = out["Pack terminal voltage [V]"]
    It = out["Pack current [A]"]
    Pk = out["Pack power [W]"]
    topo_changes = out["Topology changes [s]"]

    # ---- 支路电流与环流 ----
    # 串联支路中 8 颗电芯电流相同，取该支路任意一颗即可代表支路电流。
    # 支路 A：cell 0..7（并网前后序号一致）；支路 B：cell 8..15。
    I_A = I_cell[:, 0]                  # 支路 A 电流（放电为正）
    I_B = I_cell[:, 8]                  # 支路 B 电流（放电为正）
    # 环流（并联组间失衡电流）：两组并联，I_A + I_B = I_load。
    #   均衡时各分担 100A；失衡时高 EMF 的 B 多放、A 少放（甚至被充电）。
    #   定义环流 δ = (I_B - I_A)/2：δ>0 表示环流沿「B→A」方向（B 给 A 充电的净环流）。
    #   注意：该定义仅在并入后的并行段有意义；solo 段支路 B 电流为 0，δ 只是单支路代理值。
    I_circ = (I_B - I_A) / 2.0
    I_circ_pre = I_circ.copy()
    I_circ[t < t_solo] = np.nan          # solo 段无环流，绘图置 NaN 避免误导

    # 仅取并入后的并行段做统计分析
    mask_par = t >= t_solo
    t_p = t[mask_par]
    Icirc_p = I_circ_pre[mask_par]
    # 并入后第一个记录点（环流峰值，基本等于并入瞬间的浪涌）
    i_switch = int(np.searchsorted(t, t_solo))
    peak_circ = float(np.abs(I_circ_pre[i_switch])) if i_switch < len(I_circ_pre) else 0.0
    # 稳态环流（末段平均）
    steady_circ = float(np.mean(np.abs(I_circ_pre[-30:])))
    # 环流衰减时间常数：对 |I_circ| 做指数拟合 I(t)=a*exp(-t/tau)
    if t_p.size > 5 and np.all(np.abs(Icirc_p) > 1e-6):
        y = np.abs(Icirc_p)
        y0 = max(y[0], 1e-9)
        ln_y = np.log(y / y0)
        # 线性拟合 ln(y/y0) = -t/tau  →  tau = -1/slope
        A = np.vstack([t_p - t_p[0], np.ones_like(t_p)]).T
        slope, _ = np.linalg.lstsq(A, ln_y, rcond=None)[0]
        tau_fit = -1.0 / slope if slope < 0 else float("nan")
    else:
        tau_fit = float("nan")
    # 尾部拟合：剔除前 150s（RC 恢复主导段），提取慢速 SoC 均衡时间尺度
    tau_tail = float("nan")
    i_tail = int(np.searchsorted(t_p, t_p[0] + 150.0))
    if t_p.size - i_tail > 10 and np.all(np.abs(Icirc_p[i_tail:]) > 1e-6):
        y2 = np.abs(Icirc_p[i_tail:])
        y2_0 = max(y2[0], 1e-9)
        A2 = np.vstack([t_p[i_tail:] - t_p[i_tail], np.ones_like(t_p[i_tail:])]).T
        slope2, _ = np.linalg.lstsq(A2, np.log(y2 / y2_0), rcond=None)[0]
        tau_tail = -1.0 / slope2 if slope2 < 0 else float("nan")

    # SoC 均衡
    socA = soc[:, 0:8].mean(axis=1)
    socB = soc[:, 8:16].mean(axis=1)
    socA_pre = float(socA[i_switch])
    socB_pre = float(socB[i_switch])
    socA_post = float(socA[-1])
    socB_post = float(socB[-1])

    # ---- 文本报告 ----
    print()
    print("-" * 70)
    print(" 结果摘要")
    print("-" * 70)
    print(f" 拓扑切换时刻   : {topo_changes}")
    print(f" 并入瞬间 SoC   : 支路A={socA_pre:.4f}  支路B={socB_pre:.4f}  (Δ={socB_pre-socA_pre:.4f})")
    print(f" 并入瞬环流峰值 : {peak_circ:.2f} A  (B→A 方向)")
    print(f"   其中支路A={I_A[i_switch]:.2f}A(被充电)  支路B={I_B[i_switch]:.2f}A")
    print(f" 末段稳态环流   : {steady_circ:.3f} A")
    if not np.isnan(tau_fit):
        print(f" 环流衰减τ(全段拟合): {tau_fit:.1f} s")
    if not np.isnan(tau_tail):
        print(f" 环流衰减τ(尾部拟合): {tau_tail:.1f} s  ← 慢速SoC均衡主导时间尺度")
    print(f" 末态 SoC       : 支路A={socA_post:.4f}  支路B={socB_post:.4f}  (Δ={socB_post-socA_post:.4f})")
    print(f" 整包端电压@末态: {Vt[-1]:.3f} V   总功率={Pk[-1]/1000:.2f} kW")
    print(f" 末态支路电流   : A={I_A[-1]:.2f} A   B={I_B[-1]:.2f} A  (应≈各100A)")
    print("-" * 70)
    print()
    print(" 物理解读（环流来源 = EMF 差 / 组间环阻）：")
    print("  · 并入瞬间两支路 EMF 差≈0.84V：")
    print("      - OCV 差（B 1.0 vs A 0.9365）≈0.22V；")
    print("      - RC 极化失配占主导：A 已放电360s，极化被『耗尽』(v_rc≈−0.62V/组)，")
    print("        B 闲时待机、RC=0、OCV 全额可用 → 额外≈0.62V 差。")
    print("  · 组间环阻仅≈6.4mΩ(两组各≈3.2mΩ)，故产生≈140A 合闸涌流/环流；")
    print("  · 这是真实工程风险（接触器合闸冲击电流），BMS 通常需预充/限流缓冲。")
    print(" 环流方向：始终 B→A（高 EMF 的 B 给低 EMF 的 A 充电/减缓 A 放电，被动均衡）。")
    print("=" * 70)

    # ---- 绘图 ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("8S2P / 总负载 200A 工况  ——  环流分析\n支路A先单独工作0.1h，支路B从闲时并入并联", fontsize=14)

    # (1) 整包端电压 & 总电流
    ax[0, 0].plot(t / 60, Vt, color="#1f77b4", lw=1.6, label="端电压 [V]")
    ax[0, 0].set_ylabel("端电压 [V]", color="#1f77b4")
    ax[0, 0].tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax[0, 0].twinx()
    ax2.plot(t / 60, It, color="#d62728", lw=1.2, ls="--", label="总电流 [A]")
    ax2.set_ylabel("总电流 [A]", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax[0, 0].axvline(t_solo / 60, color="k", lw=1.0, ls=":")
    ax[0, 0].set_title("整包端口：端电压 / 总电流")
    ax[0, 0].set_xlabel("时间 [min]")
    ax[0, 0].grid(alpha=0.3)

    # (2) 两支路电流（环流的直接观测量）
    ax[0, 1].plot(t / 60, I_A, color="#1f77b4", lw=1.6, label="支路A电流 I_A")
    ax[0, 1].plot(t / 60, I_B, color="#2ca02c", lw=1.6, label="支路B电流 I_B")
    ax[0, 1].axhline(100.0, color="gray", lw=0.8, ls=":")
    ax[0, 1].text(0.2, 101.5, "均分线 100A", color="gray", fontsize=8)
    ax[0, 1].axvline(t_solo / 60, color="gray", lw=1.0, ls=":")
    ax[0, 1].set_title("两支路电流（并入前仅A工作）")
    ax[0, 1].set_xlabel("时间 [min]")
    ax[0, 1].set_ylabel("支路电流 [A]")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)

    # (3) 环流 |δ| 衰减
    ax[1, 0].plot(t / 60, np.abs(I_circ), color="#9467bd", lw=1.8,
                  label="环流 |δ| = |I_B−I_A|/2")
    ax[1, 0].axvline(t_solo / 60, color="gray", lw=1.0, ls=":")
    if not np.isnan(tau_fit):
        # 叠加指数衰减参考曲线，直观展示衰减速率
        y0f = max(np.abs(I_circ_pre[i_switch]), 1e-9)
        ref = y0f * np.exp(-(t_p - t_p[0]) / tau_fit)
        ax[1, 0].plot(t_p / 60, ref, color="#ff7f0e", lw=1.0, ls="--",
                      label=f"全段拟合 τ={tau_fit:.0f}s")
        ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("组间环流（B→A 倒灌）衰减")
    ax[1, 0].set_xlabel("时间 [min]")
    ax[1, 0].set_ylabel("环流幅值 [A]")
    ax[1, 0].grid(alpha=0.3)

    # (4) 两支路 SoC 均衡
    ax[1, 1].plot(t / 60, socA, color="#1f77b4", lw=1.6, label="支路A 平均SoC")
    ax[1, 1].plot(t / 60, socB, color="#2ca02c", lw=1.6, label="支路B 平均SoC")
    ax[1, 1].axvline(t_solo / 60, color="gray", lw=1.0, ls=":")
    ax[1, 1].set_title("两支路 SoC：并入后被动均衡")
    ax[1, 1].set_xlabel("时间 [min]")
    ax[1, 1].set_ylabel("SoC")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ecm_pack_8s2p_200a.png")
    out_png = os.path.abspath(out_png)
    fig.savefig(out_png, dpi=130)
    print(f" 图已保存: {out_png}")

    # 同时保存一份数值结果 CSV 便于核查
    out_csv = os.path.splitext(out_png)[0] + ".csv"
    np.savetxt(
        out_csv,
        np.column_stack([t, Vt, It, I_A, I_B, I_circ_pre, socA, socB]),
        delimiter=",",
        header="Time_s,Pack_V,Pack_I_A,I_branchA,I_branchB,Circulation_A,SocA,SocB",
        fmt="%.6f",
    )
    print(f" 数据已保存: {out_csv}")


if __name__ == "__main__":
    main()

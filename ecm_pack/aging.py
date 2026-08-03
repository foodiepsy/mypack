# aging.py
# 逐芯半经验老化模型（Arrhenius 日历 + 循环），用于「堆叠温度不均 -> 寿命不均」仿真。
#
# 开关语义（用户要求）：
#   Pack.solve(aging=False)                          # 默认，完全不计算，零开销
#   Pack.solve(aging=True)                           # 启用，默认参数
#   Pack.solve(aging=AgingParams(...))               # 启用，自定义参数
#   Pack.solve(aging={...})                          # 启用，dict -> AgingParams
#
# 老化状态逐芯独立维护，每步读取该芯温度 T、电流 I、SoC，更新：
#   日历老化：Q_cal(t) = A_cal · exp(-Ea_cal/(R·T)) · √(t_days)
#   循环老化：Q_cyc = Σ半循环 B·exp(-(Ea_cyc+η·C_rate)/(R·T)) · Ah^p · (DoD/DoD_ref)^k
#   容量保持率：sohc = 1 - (Q_cal + Q_cyc)        （下限 0.5）
#   内阻增长：  R_growth = 1 + cap_fade_to_R·(Q_cal+Q_cyc)
# 回喂：容量衰减 -> SoC 演化分母变小；内阻增长 -> R0 放大（欧姆热↑），
# 形成「温度 -> 老化 -> 温度/产热」的自洽闭环。
import numpy as np
from dataclasses import dataclass, field

R_GAS = 8.314  # J/(mol·K)


@dataclass
class AgingParams:
    """老化模型参数（默认值量级对标 LFP 半经验文献）。"""
    enabled: bool = False        # 开关（Pack.solve 启用时强制置 True）
    T_ref: float = 298.15        # 参考温度 [K]
    A_cal: float = 1.2e-2        # 日历老化系数（容量损失 / √day 尺度）
    Ea_cal: float = 24000.0      # 日历老化活化能 [J/mol]
    B_cyc: float = 8.0e-3        # 循环老化系数
    Ea_cyc: float = 20000.0      # 循环老化活化能 [J/mol]
    eta: float = 0.5             # C-rate 对活化能的增量系数 [J·h/(mol·A)]（经验）
    p_ah: float = 0.55           # Ah 吞吐指数（√Ah 量级）
    DoD_ref: float = 0.8         # 参考放电深度
    k_dod: float = 1.0           # DoD 幂指数
    cap_fade_to_R: float = 0.5   # 容量衰减 -> 内阻增长折算系数
    q_cal_floor: float = 0.0     # 日历老化速率下限（数值保护，可选）


class AgingState:
    """单芯老化状态机：只依赖该芯自身 (T, I, SoC) 时序，可并行/独立推进。"""

    def __init__(self, params):
        self.p = params
        self.reset()

    def reset(self):
        self.t = 0.0                 # 累计时间 [s]
        self.Ah_throughput = 0.0     # 累计 Ah 吞吐（|I| 积分）
        self.q_cal = 0.0             # 日历容量损失（分数）
        self.q_cyc = 0.0             # 循环容量损失（分数）
        self.R_growth = 0.0          # 内阻增长（绝对增量，倍数 = 1+R_growth）
        self._soc_prev = None
        self._soc_prev2 = None
        self._soc_min = None
        self._soc_max = None

    @property
    def q_loss_total(self):
        return self.q_cal + self.q_cyc

    def capacity_retention(self):
        """容量保持率 sohc（0.5 ~ 1.0）。"""
        return float(np.clip(1.0 - self.q_loss_total, 0.5, 1.0))

    def resistance_growth(self):
        """内阻增长倍数（>= 1.0）。"""
        return float(1.0 + self.p.cap_fade_to_R * self.q_loss_total)

    def update(self, Tdeg, I, soc, dt, cap_ref=314.0):
        """推进一步老化状态。
        Tdeg   : 该芯温度 [°C]
        I      : 该芯电流 [A]（未接入电芯传 0，仅日历老化继续）
        soc    : 该芯 SoC [0,1]
        dt     : 步长 [s]
        cap_ref: 该芯当前容量 [Ah]（用于 C-rate，默认 314Ah）
        """
        self.t += dt
        Tk = max(float(Tdeg) + 273.15, 200.0)  # 数值保护
        # ---- 日历老化：积分形式（随 t_days 单调增）----
        t_days = self.t / 86400.0
        self.q_cal = (
            self.p.A_cal * np.exp(-self.p.Ea_cal / (R_GAS * Tk)) * np.sqrt(t_days)
            + self.p.q_cal_floor
        )
        # ---- 循环老化：Ah 吞吐 + DoD 摆幅结算 ----
        dAh = abs(float(I)) * dt / 3600.0
        self.Ah_throughput += dAh
        if self.p.B_cyc > 0 and abs(float(I)) > 1e-9:
            C_rate = abs(float(I)) / max(float(cap_ref), 1.0)
            self._settle_half_cycle(float(soc), C_rate, Tk)
        # ---- 内阻增长（容量衰减折算）----
        self.R_growth = self.p.cap_fade_to_R * self.q_loss_total

    # ---------- 内部：DoD 摆幅检测与半循环结算 ----------
    def _settle_half_cycle(self, soc, C_rate, Tk):
        # 方向转向检测：soc 序列二阶差分变号
        if self._soc_prev2 is not None and self._soc_prev is not None:
            d1 = soc - self._soc_prev
            d2 = self._soc_prev - self._soc_prev2
            if d1 * d2 < 0:
                dod = max(
                    (self._soc_max if self._soc_max is not None else soc)
                    - (self._soc_min if self._soc_min is not None else soc),
                    0.0,
                )
                if dod > 0.02:
                    self.q_cyc += self._cyc_increment(dod, C_rate, Tk)
                # 摆幅窗口重置
                self._soc_min = soc
                self._soc_max = soc
        # 更新摆幅跟踪
        if self._soc_min is None:
            self._soc_min = self._soc_max = soc
        else:
            self._soc_min = min(self._soc_min, soc)
            self._soc_max = max(self._soc_max, soc)
        # 更新转向检测链
        self._soc_prev2 = self._soc_prev
        self._soc_prev = soc

    def _cyc_increment(self, dod, C_rate, Tk):
        rate = self.p.B_cyc * np.exp(
            -(self.p.Ea_cyc + self.p.eta * C_rate) / (R_GAS * Tk)
        )
        return (
            0.5  # 半循环
            * rate
            * (self.Ah_throughput ** self.p.p_ah)
            * ((dod / self.p.DoD_ref) ** self.p.k_dod)
        )

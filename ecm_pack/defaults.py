# defaults.py
# 工业级大容量电芯默认参数配置
#
# 314Ah 储能大电芯（参考宁德时代、亿纬等主流储能电芯规格）：
#   - 容量 314Ah，标称电压 3.2V，能量约 1.0 kWh
#   - 尺寸约 71mm(厚) × 207mm(宽) × 720mm(高)（方形铝壳）
#   - 质量约 5.4 kg，密度约 2520 kg/m³
#   - 平均比热容约 1100 J/(kg·K)
#   - 导热系数：面内 kxy≈1.5，厚度方向 kz≈0.2（叠层各向异性）
import numpy as np

from .data import lookup_1d
from .ecm import ECMCellSpec


def make_ocv_314ah():
    """314Ah 电芯 OCV 曲线（LFP 体系，平台电压特征）。"""
    s = np.linspace(0.0, 1.0, 101)
    # LFP 典型：低 SoC 陡升、中段长平台 ~3.2V、末端缓升
    v = (
        2.50
        + 0.55 * np.tanh(6.0 * (s - 0.05))   # 低 SoC 上升
        + 0.18 * (s ** 2)                       # 高 SoC 缓升
        + 0.02 * np.sin(2 * np.pi * s)          # 微小波动
    )
    return lookup_1d(s, v)


def R0_314ah(Tdeg, I, soc):
    """314Ah 大电芯欧姆内阻：~0.3 mΩ 级，随温度/SoC 变化。"""
    base = 0.00035  # 0.35 mΩ（大电芯典型）
    soc_term = 1.0 + 0.4 * (1.0 - soc)  # 低 SoC 内阻升
    arrhenius = np.exp(2500.0 * (1.0 / 298.15 - 1.0 / (Tdeg + 273.15)))
    return base * soc_term * arrhenius + 1e-7 * abs(I)


def R1_314ah(Tdeg, I, soc):
    """极化电阻 R1：~0.15 mΩ。"""
    return 0.45 * R0_314ah(Tdeg, I, soc)


def C1_314ah(Tdeg, I, soc):
    """极化电容 C1：~80000 F（大电芯 RC 时间常数 ~12s）。"""
    return 80000.0


def dUdT_314ah(Tdeg, soc):
    """熵变系数（LFP 约为 -0.2 ~ +0.1 mV/K）。"""
    return -1e-4 + 3e-4 * soc


def cell_314ah_spec(soc_init=0.5, T_init=298.15):
    """
    返回一个 314Ah 大电芯的 ECMCellSpec，已填入三维几何与热物性参数。

    几何（方形铝壳储能电芯）：
        Lx = 0.071 m (厚度)
        Ly = 0.207 m (宽度)
        Lz = 0.720 m (高度)
    热物性：
        rho = 2520 kg/m³, cp = 1100 J/(kg·K)
        k = (1.5, 1.5, 0.2) W/(m·K)  各向异性（叠层厚度方向导热差）
    """
    return ECMCellSpec(
        capacity=314.0,
        ocv=make_ocv_314ah(),
        R0=R0_314ah,
        R=[R1_314ah],
        C=[C1_314ah],
        dUdT=dUdT_314ah,
        soc_init=soc_init,
        T_init=T_init,
        # 三维热模型几何与热物性
        Lx=0.071, Ly=0.207, Lz=0.720,
        rho=2520.0, cp=1100.0,
        k=(1.5, 1.5, 0.2),  # (kx, ky, kz) 各向异性
    )

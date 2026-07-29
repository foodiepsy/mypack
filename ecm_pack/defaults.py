# defaults.py
# 工业级大容量电芯默认参数配置
#
# 314Ah 储能大电芯（用户指定参数）：
#   - 容量 314Ah，标称电压 3.2V（LFP 体系）
#   - 尺寸：宽(X) 174mm × 高(Z) 207mm × 厚(Y) 71.7mm
#   - 密度 2300 kg/m³，比热容 1000 J/(kg·K)
#   - 导热系数（各向异性）：
#       X(宽度) 12 W/mK, Y(厚度) 0.7 W/mK, Z(高度) 11.6 W/mK
#   - 内阻 R0 = 0.4 mΩ, R1 = 0.4 mΩ, τ = R1·C1 = 100s
import numpy as np

from .data import lookup_1d
from .ecm import ECMCellSpec


def make_ocv_314ah():
    """314Ah 电芯 OCV 曲线（LFP 体系，平台电压特征）。"""
    s = np.linspace(0.0, 1.0, 101)
    v = (
        2.50
        + 0.55 * np.tanh(6.0 * (s - 0.05))
        + 0.18 * (s ** 2)
        + 0.02 * np.sin(2 * np.pi * s)
    )
    return lookup_1d(s, v)


# ---- 用户指定参数 ----
R0_DEFAULT = 0.4e-3        # 0.4 mΩ
R1_DEFAULT = 0.4e-3        # 0.4 mΩ
TAU_DEFAULT = 100.0        # R1·C1 = 100s -> C1 = tau/R1 = 250000 F
C1_DEFAULT = TAU_DEFAULT / R1_DEFAULT  # 250000 F

# 几何（用户指定）：宽174 高207 厚71.7 mm
# 约定：X=宽度, Y=厚度, Z=高度
LX_DEFAULT = 0.174         # 宽度 [m]
LY_DEFAULT = 0.0717        # 厚度 [m]
LZ_DEFAULT = 0.207         # 高度 [m]

# 热物性（用户指定）
RHO_DEFAULT = 2300.0       # kg/m³
CP_DEFAULT = 1000.0        # J/(kg·K)
K_DEFAULT = (12.0, 0.7, 11.6)  # (kx=宽度, ky=厚度, kz=高度) W/(m·K)


def R0_314ah(Tdeg, I, soc):
    """314Ah 大电芯欧姆内阻：0.4 mΩ，随温度/SoC 变化。

    温度依赖：电池内阻随温升而**下降**（离子电导率↑），故 Arrhenius
    指数取负号 —— exp(-Ea*(1/T_ref - 1/T))。当前为初始参数化阶段，
    后续将以 R0-SOC-T 查表形式整体替换。
    """
    base = R0_DEFAULT
    soc_term = 1.0 + 0.4 * (1.0 - soc)
    arrhenius = np.exp(-2500.0 * (1.0 / 298.15 - 1.0 / (Tdeg + 273.15)))
    return base * soc_term * arrhenius + 1e-7 * abs(I)


def R1_314ah(Tdeg, I, soc):
    """极化电阻 R1：0.4 mΩ。温度符号同 R0（随温升下降）。"""
    return R1_DEFAULT * (1.0 + 0.3 * (1.0 - soc)) * np.exp(
        -2000.0 * (1.0 / 298.15 - 1.0 / (Tdeg + 273.15))
    ) + 1e-7 * abs(I)


def C1_314ah(Tdeg, I, soc):
    """极化电容 C1：使 τ = R1·C1 ≈ 100s。"""
    return C1_DEFAULT


def dUdT_314ah(Tdeg, soc):
    """熵变系数（LFP 约为 -0.2 ~ +0.1 mV/K）。"""
    return -1e-4 + 3e-4 * soc


def cell_314ah_spec(soc_init=0.5, T_init=298.15):
    """
    返回一个 314Ah 大电芯的 ECMCellSpec，已填入三维几何与热物性参数。

    几何（用户指定）：宽(X) 174mm × 厚(Y) 71.7mm × 高(Z) 207mm
    热物性（用户指定）：
        rho = 2300 kg/m³, cp = 1000 J/(kg·K)
        k = (12, 0.7, 11.6) W/(m·K)  各向异性
            X(宽度)=12, Y(厚度)=0.7, Z(高度)=11.6
    电气（用户指定）：
        R0 = 0.4 mΩ, R1 = 0.4 mΩ, τ = 100s
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
        Lx=LX_DEFAULT, Ly=LY_DEFAULT, Lz=LZ_DEFAULT,
        rho=RHO_DEFAULT, cp=CP_DEFAULT,
        k=K_DEFAULT,
    )

# 热网络单元测试
import numpy as np
import pytest

import ecm_pack as ep


def test_implicit_euler_stable_for_large_conduction():
    """强导热 + 大步长下隐式欧拉不发散（缺陷回归）。"""
    th = ep.ThermalNetwork(2, C_th=300.0, h=0.5, T_amb=298.15,
                           conduction=[(0, 1, 30.0)], T_init=298.15)
    # 给 cell0 持续大产热，dt=10s（显式会发散的组合）
    Q = np.array([50.0, 0.0])
    for _ in range(100):
        th.step(Q, dt=10.0, t=0.0)
    assert np.all(np.isfinite(th.T)), "温度必须有限(不发散)"
    # 强耦合下两芯温度应被拉到接近
    assert abs(th.T[0] - th.T[1]) < 2.0


def test_temperature_relaxes_to_ambient():
    """无产热、无导热时温度应收敛到环境温度。"""
    th = ep.ThermalNetwork(1, C_th=100.0, h=1.0, T_amb=298.15, T_init=320.0)
    for _ in range(1000):
        th.step(np.array([0.0]), dt=1.0, t=0.0)
    assert abs(th.T[0] - 298.15) < 0.1


def test_conduction_equalizes_two_cells():
    """两芯初始温差，强导热下应趋于相等。"""
    th = ep.ThermalNetwork(2, C_th=100.0, h=0.0,
                           conduction=[(0, 1, 100.0)], T_init=300.0)
    th.set_temperature(np.array([310.0, 290.0]))
    for _ in range(500):
        th.step(np.array([0.0, 0.0]), dt=0.5, t=0.0)
    assert abs(th.T[0] - th.T[1]) < 0.1
    # 无对流时总热量守恒 -> 均值应回到初值 300
    assert abs(th.T.mean() - 300.0) < 0.1


def test_interface_resistance_basic():
    """interface_resistance 自动换算 G=A/R_th 并生效。"""
    A = 0.174 * 0.207  # X*Z 面，~0.036 m²
    th = ep.ThermalNetwork(2, C_th=100.0, h=0.0,
                           interface_resistance=[(0, 1, 0.01, A)],
                           T_init=298.15)
    # R_th=0.01 K·m²/W, A≈0.036 → G≈3.6 W/K
    # 检查导热矩阵有非对角元素
    assert abs(th.K[0, 1] - (A / 0.01)) < 1e-9
    # 推一步确认矩阵正定可用
    th.set_temperature(np.array([310.0, 298.15]))
    T = th.step(np.array([0.0, 0.0]), dt=10.0)
    assert np.all(np.isfinite(T))


def test_interface_resistance_combined_with_conduction():
    """interface_resistance 与 conduction 共存时应叠加。"""
    A = 0.01
    th = ep.ThermalNetwork(2, C_th=100.0, h=0.0,
                           conduction=[(0, 1, 2.0)],
                           interface_resistance=[(0, 1, 0.02, A)],
                           T_init=298.15)
    # G_conduction=2.0, G_ifa=A/R_th=0.01/0.02=0.5 → 总 G=2.5
    assert abs(th.K[0, 1] - 2.5) < 1e-9


def test_interface_resistance_large_blocks_heat():
    """大 R_th 应有效阻碍导热，温度梯度更大。"""
    A = 0.01
    # 小 R_th
    th_low = ep.ThermalNetwork(2, C_th=100.0, h=0.0,
                               interface_resistance=[(0, 1, 0.001, A)],
                               T_init=298.15)
    th_low.set_temperature(np.array([310.0, 298.15]))
    for _ in range(50):
        th_low.step(np.array([0.0, 0.0]), dt=1.0)
    dT_low = abs(th_low.T[0] - th_low.T[1])

    # 大 R_th（100x）
    th_high = ep.ThermalNetwork(2, C_th=100.0, h=0.0,
                                interface_resistance=[(0, 1, 0.1, A)],
                                T_init=298.15)
    th_high.set_temperature(np.array([310.0, 298.15]))
    for _ in range(50):
        th_high.step(np.array([0.0, 0.0]), dt=1.0)
    dT_high = abs(th_high.T[0] - th_high.T[1])

    # 大 R_th 的热接触更差 → 温差更大
    assert dT_high > dT_low, f"大R_th温差({dT_high:.4f})应 > 小R_th温差({dT_low:.4f})"


def test_interface_resistance_missing_area_raises():
    """未提供接触面积应报 ValueError。"""
    with pytest.raises(ValueError, match="接触面积"):
        ep.ThermalNetwork(2, C_th=100.0,
                          interface_resistance=[(0, 1, 0.01)])

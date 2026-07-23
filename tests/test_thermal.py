# 热网络单元测试
import numpy as np

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

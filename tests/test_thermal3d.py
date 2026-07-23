# 三维热模型单元测试
import numpy as np
import pytest

from ecm_pack.thermal3d import Cell3DThermal


def test_3d_step_is_finite_and_positive():
    """单步产热后温度有限且高于初温。"""
    th = Cell3DThermal(0.071, 0.207, 0.720, nx=3, ny=3, nz=4,
                       rho=2520.0, cp=1100.0, k=(1.5, 1.5, 0.2),
                       h=8.0, T_amb=298.15, T_init=298.15)
    T = th.step(100.0, 2.0)
    assert np.all(np.isfinite(T))
    assert np.all(T >= 298.15 - 1e-6)  # 不低于初温/环境


def test_3d_no_heat_relaxes_to_ambient():
    """无产热、长时间后温度收敛到环境温度。"""
    th = Cell3DThermal(0.05, 0.05, 0.05, nx=3, ny=3, nz=3,
                       rho=2000.0, cp=800.0, k=2.0,
                       h=20.0, T_amb=298.15, T_init=350.0)
    for _ in range(20000):
        th.step(0.0, 1.0)
    assert abs(th.T_avg - 298.15) < 1.0


def test_3d_single_node_matches_analytic():
    """2x2x2 网格单步与解析解一致（角点 3 面对流）。"""
    th = Cell3DThermal(0.071, 0.207, 0.720, nx=2, ny=2, nz=2,
                       rho=2520.0, cp=1100.0, k=(1.5, 1.5, 0.2),
                       h=8.0, T_amb=298.15, T_init=320.0)
    dt = 1.0
    th.step(0.0, dt)
    dx, dy, dz = th.dx, th.dy, th.dz
    Ax = dy * dz; Ay = dx * dz; Az = dx * dy
    rho_cp_V = 2520.0 * 1100.0 * th._V_cell
    hA = 8.0 * (Ax + Ay + Az)  # 角点 3 面
    T_analytic = (rho_cp_V * 320.0 + dt * hA * 298.15) / (rho_cp_V + dt * hA)
    assert abs(th.T[0] - T_analytic) < 1e-10


def test_3d_higher_heat_higher_temp():
    """产热越大温度越高（单调性）。"""
    def run(Q):
        th = Cell3DThermal(0.071, 0.207, 0.720, nx=3, ny=3, nz=4,
                           rho=2520.0, cp=1100.0, k=1.5,
                           h=8.0, T_amb=298.15, T_init=298.15)
        for _ in range(100):
            th.step(Q, 2.0)
        return th.T_avg
    T_low = run(50.0)
    T_high = run(200.0)
    assert T_high > T_low


def test_3d_temperature_stats():
    """temperature_stats 返回正确的字段与值。"""
    th = Cell3DThermal(0.071, 0.207, 0.720, nx=3, ny=3, nz=4,
                       rho=2520.0, cp=1100.0, k=1.5,
                       h=8.0, T_amb=298.15, T_init=298.15)
    th.step(100.0, 2.0)
    stats = th.temperature_stats()
    assert set(stats.keys()) == {"T_max [K]", "T_min [K]", "T_avg [K]", "dT_max [K]"}
    assert stats["T_max [K]"] >= stats["T_min [K]"]
    assert abs(stats["dT_max [K]"] - (stats["T_max [K]"] - stats["T_min [K]"])) < 1e-9


def test_314ah_default_spec_has_geometry():
    """314Ah 默认规格已填入三维几何与热物性。"""
    import ecm_pack as ep
    spec = ep.cell_314ah_spec()
    assert spec.capacity == 314.0
    assert spec.Lx is not None and spec.Ly is not None and spec.Lz is not None
    assert spec.rho is not None and spec.cp is not None
    assert spec.k is not None
    # 体积合理（约 10L）
    vol = spec.Lx * spec.Ly * spec.Lz
    assert 0.005 < vol < 0.02

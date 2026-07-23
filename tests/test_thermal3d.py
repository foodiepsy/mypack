# 多维热模型单元测试（1D/2D/3D）
import numpy as np
import pytest

from ecm_pack.thermal3d import CellThermalModel, Cell3DThermal


def test_3d_step_is_finite_and_positive():
    """单步产热后温度有限且高于初温。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=3, ny=3, nz=4,
                          rho=2300.0, cp=1000.0, k=(12, 0.7, 11.6),
                          h=8.0, T_amb=298.15, T_init=298.15)
    T = th.step(100.0, 2.0)
    assert np.all(np.isfinite(T))
    assert np.all(T >= 298.15 - 1e-6)


def test_1d_only_x_direction():
    """1D 模型：ny=nz=1，N=nx，仅沿 X 方向有导热。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=1, nx=10,
                          rho=2300.0, cp=1000.0, k=12.0,
                          h=8.0, T_amb=298.15, T_init=298.15)
    assert th.ny == 1 and th.nz == 1
    assert th.N == 10
    T = th.step(100.0, 2.0)
    assert np.all(np.isfinite(T))
    assert T.shape == (10,)


def test_2d_xy_plane():
    """2D 模型：nz=1，N=nx*ny，XY 平面求解。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=2, nx=5, ny=4,
                          rho=2300.0, cp=1000.0, k=(12, 0.7, 11.6),
                          h=8.0, T_amb=298.15, T_init=298.15)
    assert th.nz == 1
    assert th.N == 20
    T = th.step(100.0, 2.0)
    assert T.shape == (20,)


def test_no_heat_relaxes_to_ambient():
    """无产热、长时间后温度收敛到环境温度。"""
    th = CellThermalModel(0.05, 0.05, 0.05, dim=3, nx=3, ny=3, nz=3,
                          rho=2000.0, cp=800.0, k=2.0,
                          h=20.0, T_amb=298.15, T_init=350.0)
    for _ in range(20000):
        th.step(0.0, 1.0)
    assert abs(th.T_avg - 298.15) < 1.5


def test_single_node_matches_analytic():
    """2x2x2 网格单步与解析解一致（角点 3 面对流）。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=2, ny=2, nz=2,
                          rho=2300.0, cp=1000.0, k=(12, 0.7, 11.6),
                          h=8.0, T_amb=298.15, T_init=320.0)
    dt = 1.0
    th.step(0.0, dt)
    dx, dy, dz = th.dx, th.dy, th.dz
    Ax = dy * dz; Ay = dx * dz; Az = dx * dy
    rho_cp_V = 2300.0 * 1000.0 * th._V_cell
    hA = 8.0 * (Ax + Ay + Az)
    T_analytic = (rho_cp_V * 320.0 + dt * hA * 298.15) / (rho_cp_V + dt * hA)
    assert abs(th.T[0] - T_analytic) < 1e-10


def test_higher_heat_higher_temp():
    """产热越大温度越高（单调性）。"""
    def run(Q):
        th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=3, ny=3, nz=4,
                              rho=2300.0, cp=1000.0, k=1.5,
                              h=8.0, T_amb=298.15, T_init=298.15)
        for _ in range(100):
            th.step(Q, 2.0)
        return th.T_avg
    assert run(200.0) > run(50.0)


def test_temperature_stats():
    """temperature_stats 返回正确字段。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=3, ny=3, nz=4,
                          rho=2300.0, cp=1000.0, k=1.5,
                          h=8.0, T_amb=298.15, T_init=298.15)
    th.step(100.0, 2.0)
    stats = th.temperature_stats()
    assert set(stats.keys()) == {"T_max [K]", "T_min [K]", "T_avg [K]", "dT_max [K]"}
    assert stats["T_max [K]"] >= stats["T_min [K]"]


def test_314ah_default_spec_has_geometry():
    """314Ah 默认规格已填入用户指定的几何与热物性。"""
    import ecm_pack as ep
    spec = ep.cell_314ah_spec()
    assert spec.capacity == 314.0
    assert abs(spec.Lx - 0.174) < 1e-9   # 宽度
    assert abs(spec.Ly - 0.0717) < 1e-9  # 厚度
    assert abs(spec.Lz - 0.207) < 1e-9   # 高度
    assert spec.rho == 2300.0
    assert spec.cp == 1000.0
    assert spec.k == (12.0, 0.7, 11.6)


def test_backward_compat_alias():
    """Cell3DThermal 是 CellThermalModel 的别名。"""
    assert Cell3DThermal is CellThermalModel


def test_anisotropic_k_affects_temperature():
    """各向异性导热：ky(厚度0.7)远小于kx(宽度12)，厚度方向温差应更大。"""
    # 沿 Y 方向施加大网格，看温差
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=3, ny=5, nz=3,
                          rho=2300.0, cp=1000.0, k=(12, 0.7, 11.6),
                          h=8.0, T_amb=298.15, T_init=298.15)
    for _ in range(200):
        th.step(500.0, 2.0)
    stats = th.temperature_stats()
    assert stats["dT_max [K]"] > 0.01  # 有可测温差

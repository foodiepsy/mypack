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


# ───────────────────── 非对称冷却（面差异化 h）─────────────────────

def test_uniform_h_backward_compat():
    """标量 h 仍与旧行为完全一致（向后兼容）。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=3, ny=3, nz=4,
                          rho=2300.0, cp=1000.0, k=(12, 0.7, 11.6),
                          h=8.0, T_amb=298.15, T_init=298.15)
    T = th.step(100.0, 2.0)
    assert np.all(np.isfinite(T))
    assert np.all(T >= 298.15 - 1e-9)


def test_tuple_h_six_faces():
    """6 元组 (hx0,hx1,hy0,hy1,hz0,hz1) 各面独立 h。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=3, ny=3, nz=4,
                          rho=2300.0, cp=1000.0, k=(12, 0.7, 11.6),
                          h=(200.0, 5.0, 5.0, 5.0, 5.0, 5.0),  # x0 强冷
                          T_amb=298.15, T_init=298.15)
    assert th.h_faces == (200.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    T = th.step(100.0, 2.0)
    assert np.all(np.isfinite(T))


def test_dict_h_with_default():
    """Dict 格式 h，'default' 键回退指定未列出的面。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=3, ny=3, nz=4,
                          rho=2300.0, cp=1000.0, k=1.5,
                          h={"default": 5.0, "x0": 200.0},
                          T_amb=298.15)
    assert th.h_faces == (200.0, 5.0, 5.0, 5.0, 5.0, 5.0)
    T = th.step(0.0, 60.0)  # 零产热，应趋向 T_amb
    assert np.all(np.isfinite(T))


def test_asymmetric_cooling_cold_face_colder():
    """非对称冷却：强冷 x0 面一侧温度应低于对面 x1 侧。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=1, nx=20,
                          rho=2300.0, cp=1000.0, k=1.5,
                          h=(200.0, 5.0, 5.0, 5.0, 5.0, 5.0),
                          T_amb=298.15, T_init=298.15)
    for _ in range(100):
        th.step(1000.0, 1.0)  # 持续产热
    T1d = th.T  # shape (20,) → x0 在 index 0, x1 在 index -1
    assert T1d[0] < T1d[-1], f"强冷端(T_x0={T1d[0]:.3f})应低于弱冷端(T_x1={T1d[-1]:.3f})"


def test_h_parse_errors():
    """非法 h 格式应抛出 ValueError/TypeError。"""
    # 元组长度错误
    with pytest.raises(ValueError, match="6"):
        CellThermalModel(0.1, 0.1, 0.1, h=(1.0, 2.0, 3.0))
    # 无效类型
    with pytest.raises(TypeError):
        CellThermalModel(0.1, 0.1, 0.1, h="invalid")


def test_h_faces_property():
    """h_faces 属性返回准确的六元组。"""
    th = CellThermalModel(0.1, 0.1, 0.1, h={"x0": 10, "y1": 20})
    assert th.h_faces == (10.0, 5.0, 5.0, 20.0, 5.0, 5.0)


# ───────────────────── 壳层热阻（表面温度子模型）─────────────────────

def test_r_shell_zero_means_no_shell():
    """R_shell=0 时 T_surface = T_avg（默认行为）。"""
    th = CellThermalModel(0.1, 0.1, 0.1, dim=1, nx=5,
                          rho=2300.0, cp=1000.0, k=1.5,
                          h=8.0, T_amb=298.15, T_init=310.0,
                          R_shell=0.0)
    th.step(0.0, 10.0)
    assert abs(th.T_surface - th.T_avg) < 1e-9


def test_r_shell_positive_creates_gap():
    """R_shell>0 时 T_surface 应介于 T_avg 与 T_amb 之间。"""
    th = CellThermalModel(0.1, 0.1, 0.1, dim=1, nx=5,
                          rho=2300.0, cp=1000.0, k=1.5,
                          h=8.0, T_amb=298.15, T_init=310.0,
                          R_shell=0.5)  # 0.5 K/W 壳层热阻
    for _ in range(20):
        th.step(0.0, 10.0)  # 零产热，向环境散热
    # T_init=310, T_amb=298.15, 散热中 T_avg 应 < 310，但 T_surface 应更接近 T_amb
    assert th.T_surface < th.T_avg, "壳层温度应低于体平均温度"
    assert th.T_surface > 298.15, "壳层温度应高于环境（处于散热中）"
    # T_surface 应介于体温和环境之间
    assert th.T_surface > 298.15 and th.T_surface < th.T_avg


def test_r_shell_in_stats():
    """temperature_stats 应包含 T_surface 和 T_core_max。"""
    th = CellThermalModel(0.1, 0.1, 0.1, dim=1, nx=5,
                          h=8.0, T_amb=298.15, T_init=310.0,
                          R_shell=0.2)
    th.step(10.0, 10.0)
    s = th.temperature_stats()
    assert "T_surface [K]" in s
    assert "T_core_max [K]" in s


def test_r_shell_larger_gap():
    """R_shell 越大，T_core 与 T_surface 的温差越大。"""
    def _delta(Rs):
        th = CellThermalModel(0.1, 0.1, 0.1, dim=1, nx=5,
                              h=8.0, T_amb=298.15, T_init=310.0,
                              R_shell=Rs)
        for _ in range(10):
            th.step(0.0, 10.0)
        return th.T_avg - th.T_surface

    dt_small = _delta(0.1)
    dt_large = _delta(1.0)
    assert dt_large > dt_small, f"大R_shell温差({dt_large:.4f})应>小R_shell温差({dt_small:.4f})"


# ───────────────────── 热场可视化 ─────────────────────

def test_plot_slice_1d():
    """1D 模型的 plot_slice 返回 figure。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=1, nx=10,
                          h=5.0, T_amb=298.15, T_init=310.0)
    th.step(100.0, 2.0)
    fig, ax = th.plot_slice()
    assert fig is not None
    import matplotlib; matplotlib.pyplot.close(fig)


def test_plot_slice_3d():
    """3D 模型的 plot_slice 三平面均可绘制。"""
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=3, nx=4, ny=4, nz=5,
                          h=5.0, T_amb=298.15, T_init=310.0)
    th.step(100.0, 2.0)
    for plane in ("xy", "xz", "yz"):
        fig, ax = th.plot_slice(plane=plane)
        assert fig is not None
        import matplotlib; matplotlib.pyplot.close(fig)


def test_plot_summary_save():
    """plot_summary 可保存到文件。"""
    import tempfile, os
    th = CellThermalModel(0.174, 0.0717, 0.207, dim=2, nx=5, ny=5,
                          h=5.0, T_amb=298.15, T_init=310.0)
    th.step(100.0, 2.0)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        fig = th.plot_summary(save_path=path)
        import matplotlib; matplotlib.pyplot.close(fig)
        assert os.path.getsize(path) > 100
    finally:
        os.unlink(path)

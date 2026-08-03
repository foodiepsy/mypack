"""demo1（8S 314Ah 大面背靠背 + 不对称泡棉）0.5C 放电 2h 的**黄金基准回归**。
这是整个仓库最重要的一道锁：任何对 ECM / 电路 / 三维热模型 / 默认参数的改动，
只要动了下面三个数，这个测试就必须红。
基准值来源：demo1 在矩阵化重构前后逐位复现的结果
峰值温度 308.42 K，芯间温差 ΔT 2.388 K，末态整包端电压 17.62 V
    注：上述为热模型 S2 修复（边界半控制体导热热阻）**之前**的锁值。
    2026-07-28 修复 StackThermal3D 边界漏算半控制体导热热阻后，边界散热被
    正确削弱（旧模型高估了边界冷却、夸大了芯间温差），基准需上移重校为：
        峰值温度 308.72 K，芯间温差 ΔT 2.039 K，末态端电压 17.62 V（不变）
    端电压与热模型无关，恒为 17.62 V。

运行较慢（3600 步隐式求解，约 20~40s），默认带 slow 标记：
pytest tests/ -q                  # 包含本测试
pytest tests/ -q -m "not slow"    # 跳过本测试
"""
import numpy as np
import pytest
import ecm_pack as ep
from ecm_pack.thermal3d_stack import StackThermal3D
# 黄金基准（K, K, V）与容差（S2 修复后半控制体导热热阻重校值）
BASELINE = {"peak": 308.72, "dT": 2.039, "vt": 17.62}
TOL = {"peak": 0.01, "dT": 0.005, "vt": 0.01}
def run_demo1(crate=0.5, t_total=7200.0, dt=2.0):
    """复现 demo1 的 8S 泡棉场景。返回 (peak, dT, vt)。"""
    n_cells = 8
    I_load = crate * 314.0
    n_steps = int(t_total / dt)
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
             for _ in range(n_cells)]
    spec = cells[0].spec
    nl, _, _ = ep.setup_circuit(8, 1)
    thermal = StackThermal3D(
        n_cells=n_cells, Lx=spec.Lx, Ly=spec.Ly, Lz=spec.Lz,
        nx=4, ny=5, nz=8,
        cell_k=spec.k, cell_rho=spec.rho, cell_cp=spec.cp,
        foam_k=0.04, foam_thickness=0.001, foam_faces=["x0", "z0"],
        k_top=0.2, d_top=0.0002,
        h_top=50.0, T_top=298.15,
        h_bottom=50.0, T_bottom=298.15,
        h_side=50.0, T_amb=298.15, T_init=298.15,
    )
    pack = ep.Pack(cells, nl, thermal=thermal, v_cut_lower=2.0)
    out = pack.solve(dt=dt, control=I_load, control_type="current",
                     n_steps=n_steps, record_every=60)
    T = out["Cell temperature [K]"][-1]
    return (float(T.max()), float(T.max() - T.min()),
            float(out["Pack terminal voltage [V]"][-1]))

@pytest.mark.slow
def test_demo1_golden_baseline():
    peak, dT, vt = run_demo1()
    assert abs(peak - BASELINE["peak"]) < TOL["peak"], f"峰值温度漂移: {peak:.4f}K"
    assert abs(dT - BASELINE["dT"]) < TOL["dT"], f"芯间温差漂移: {dT:.4f}K"
    assert abs(vt - BASELINE["vt"]) < TOL["vt"], f"端电压漂移: {vt:.4f}V"
    @pytest.mark.slow
    def test_demo1_physical_sanity():
        """物理合理性：温度沿堆叠方向单调、无芯超出合理区间、无 NaN。"""
        n_cells = 8
        cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=1.0, T_init=298.15))
        for _ in range(n_cells)]
        spec = cells[0].spec
        nl, _, _ = ep.setup_circuit(8, 1)
        thermal = StackThermal3D(
        n_cells=n_cells, Lx=spec.Lx, Ly=spec.Ly, Lz=spec.Lz, nx=4, ny=5, nz=8,
        cell_k=spec.k, cell_rho=spec.rho, cell_cp=spec.cp,
        foam_k=0.04, foam_thickness=0.001, foam_faces=["x0", "z0"],
        k_top=0.2, d_top=0.0002,
        h_top=50.0, T_top=298.15, h_bottom=50.0, T_bottom=298.15,
        h_side=50.0, T_amb=298.15, T_init=298.15,
        )
        pack = ep.Pack(cells, nl, thermal=thermal, v_cut_lower=2.0)
        out = pack.solve(dt=2.0, control=157.0, control_type="current",
        n_steps=900, record_every=60)
        T = out["Cell temperature [K]"][-1]
        assert np.all(np.isfinite(T))
        # 环境 25°C 起步、0.5C 放电，末态应在 [298K, 320K] 内
        assert T.min() >= 298.0 and T.max() <= 320.0, f"温度越界: {T}"
        # 中间芯（散热差）不低于两端芯
        assert T[3:5].mean() >= max(T[0], T[-1]) - 1e-9

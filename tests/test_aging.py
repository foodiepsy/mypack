"""老化/寿命开关 + 电气参数温度双路径 的测试。
覆盖：
  1. temp_aware 三路分发（标量->经验式 / 表->查表 / 可调用->原样）
  2. 容量-温度经验修正（低温折损、25°C 恒等）
  3. R0 查表路径与解析式一致性（spec 级）
  4. ECMCellSpec 无工厂时标量退化为常数（向后兼容）
  5. Pack.solve aging 开关：默认关闭零开销、开启后有老化输出
  6. 老化启用时容量保持率单调下降、内阻增长单调上升
  7. 老化回喂影响 SoC 演化（容量衰减 -> 同电流 SoC 掉更快）
"""
import numpy as np
import pytest
import ecm_pack as ep
from ecm_pack.aging import AgingParams, AgingState


def make_pack(n=4, soc_init=1.0):
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=soc_init)) for _ in range(n)]
    nl, _, _ = ep.setup_circuit(1, n)
    return ep.Pack(cells, nl), cells


# ---------- 1. temp_aware 三路分发 ----------
def test_temp_aware_scalar_uses_empirical():
    spec = ep.cell_314ah_spec()
    # 标量 capacity -> 经验式：低温折损
    assert spec.capacity_fn(25.0, 0, 1.0) == pytest.approx(314.0)
    assert spec.capacity_fn(0.0, 0, 1.0) < 314.0
    assert spec.capacity_fn(45.0, 0, 1.0) > 314.0


def test_temp_aware_table_reads_table():
    table = ep.build_r0_soc_t_table()
    spec = ep.cell_314ah_spec(R0=table)          # 表路径
    spec_ref = ep.cell_314ah_spec()              # 经验式路径
    for Tdeg in (-10.0, 0.0, 25.0, 45.0):
        assert spec.R0(Tdeg, 5.0, 0.4) == pytest.approx(
            spec_ref.R0(Tdeg, 5.0, 0.4), abs=2e-5)


def test_temp_aware_callable_passthrough():
    def my_r0(Tdeg, I, soc):
        return 1.0e-3 + Tdeg * 1e-6
    spec = ep.cell_314ah_spec(R0=my_r0)
    assert spec.R0(35.0, 1.0, 0.5) == pytest.approx(1.0e-3 + 35e-6)


def test_cell_spec_scalar_without_factory_stays_constant():
    """ECMCellSpec 直接传标量且无 empirical_factories -> 常数（向后兼容）。"""
    spec = ep.ECMCellSpec(capacity=100.0, ocv=lambda s: 3.2, R0=5e-3)
    assert spec.capacity == 100.0
    assert spec.capacity_fn(0.0, 0, 0.5) == 100.0
    assert spec.R0(0.0, 0, 0.5) == 5e-3
    assert spec.R0(50.0, 0, 0.5) == 5e-3


# ---------- 2. aging 开关 ----------
def test_aging_default_off_zero_overhead():
    pack, _ = make_pack()
    out = pack.solve(dt=10.0, control=20.0, n_steps=30)
    assert "Cell capacity retention" not in out
    assert "Aging Q loss" not in out
    # 回喂系数保持默认
    for c in pack.cells:
        assert c.sohc == 1.0 and c.R0_growth == 1.0


def test_aging_on_emits_outputs():
    pack, _ = make_pack()
    out = pack.solve(dt=10.0, control=20.0, n_steps=30, aging=True)
    for key in ["Cell capacity retention", "Cell resistance growth",
                "Aging Ah throughput", "Aging Q loss"]:
        assert key in out
        assert out[key].shape[1] == pack.N
    assert np.all(out["Cell capacity retention"][0] == 1.0)
    assert np.all(out["Cell capacity retention"][-1] <= 1.0)
    assert np.all(out["Aging Ah throughput"][-1] > 0.0)


def test_aging_params_dict_form():
    pack, _ = make_pack()
    out = pack.solve(dt=10.0, control=20.0, n_steps=30,
                     aging={"Ea_cal": 30000.0, "B_cyc": 0.0})
    assert "Aging Q loss" in out
    # B_cyc=0 -> 只有日历老化，无循环成分；Ea_cal 更大 -> 衰减更慢
    q = out["Aging Q loss"][-1, 0]
    assert 0.0 < q < 1e-3


def test_aging_monotonic_degradation():
    params = AgingParams(A_cal=1.2e-2 * 1e6, B_cyc=8e-3 * 1e6)  # 放大 1e6 加速
    pack, _ = make_pack()
    out = pack.solve(dt=100.0, control=157.0, n_steps=300, aging=params)
    cr = out["Cell capacity retention"][:, 0]
    rg = out["Cell resistance growth"][:, 0]
    assert np.all(np.diff(cr) <= 0.0)          # 容量保持率单调不升
    assert np.all(np.diff(rg) >= 0.0)          # 内阻增长单调不降
    assert cr[-1] < cr[0]
    assert rg[-1] > rg[0]


def test_aging_feedback_speeds_up_soc_drop():
    """老化回喂：容量保持率下降 -> 同电流下 SoC 掉得更快。"""
    params = AgingParams(A_cal=1.2e-2 * 1e6, B_cyc=8e-3 * 1e6)
    # 对照组：不开老化
    pack0, cells0 = make_pack()
    out0 = pack0.solve(dt=100.0, control=157.0, n_steps=200)
    # 实验组：开老化
    pack1, cells1 = make_pack()
    out1 = pack1.solve(dt=100.0, control=157.0, n_steps=200, aging=params)
    assert out1["Cell SoC"][-1, 0] < out0["Cell SoC"][-1, 0]
    assert out1["Cell capacity retention"][-1, 0] < 1.0


# ---------- 3. AgingState 单元 ----------
def test_aging_state_no_current_only_calendar():
    st = AgingState(AgingParams())
    for _ in range(10):
        st.update(25.0, 0.0, 0.5, 3600.0)
    assert st.Ah_throughput == 0.0
    assert st.q_cyc == 0.0
    assert st.q_cal > 0.0
    assert st.capacity_retention() < 1.0

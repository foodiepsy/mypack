# 电路层(MNA)单元测试
import numpy as np
import pytest

import ecm_pack as ep
from ecm_pack.circuit import Netlist, solve_circuit, setup_circuit, setup_two_group, setup_series_bypass


def _nl(elements):
    return Netlist(elements)


# ---------- 单芯电流控制 ----------
def test_single_cell_current_control():
    """单芯 1A 放电：端电压 = E - I·R0。"""
    E, R0 = 3.9, 0.05
    nl = _nl([
        {"desc": "V0", "node1": 1, "node2": 2, "value": E},
        {"desc": "R00", "node1": 2, "node2": 0, "value": R0},
        {"desc": "I", "node1": 1, "node2": 0, "value": 0.0},
    ])
    Vn, Ib, It, Vt, Pt = solve_circuit(nl, current=1.0)
    assert abs(Vt[0] - (E - 1.0 * R0)) < 1e-9
    assert abs(It[0] - 1.0) < 1e-9


# ---------- 并联环流(缺陷回归) ----------
def test_parallel_circulation_direction():
    """两芯并联无负载：高 SoC(高电压)放电、低 SoC(低电压)被充电。"""
    nl = _nl([
        {"desc": "V0", "node1": 1, "node2": 2, "value": 4.0},
        {"desc": "R00", "node1": 2, "node2": 0, "value": 0.02},
        {"desc": "V1", "node1": 1, "node2": 3, "value": 3.0},
        {"desc": "R01", "node1": 3, "node2": 0, "value": 0.02},
        {"desc": "I", "node1": 1, "node2": 0, "value": 0.0},
    ])
    Vn, Ib, It, Vt, Pt = solve_circuit(nl, current=0.0)
    # cell_current = -I_batt：放电为正、充电为负
    c0 = -Ib[0]  # 高压芯
    c1 = -Ib[1]  # 低压芯
    assert c0 > 0, "高电压芯应放电"
    assert c1 < 0, "低电压芯应被充电"
    assert abs(c0 + c1) < 1e-9, "无负载时两芯电流应大小相等方向相反(纯环流)"
    assert abs(It[0]) < 1e-9


# ---------- 串联电压叠加 ----------
def test_series_voltage_sum():
    """2 串联同芯：端口电压 = 2·单芯电压，电流 = 负载电流。"""
    E, R0 = 3.5, 0.02
    nl = _nl([
        {"desc": "V0", "node1": 1, "node2": 3, "value": E},
        {"desc": "R00", "node1": 3, "node2": 0, "value": R0},
        {"desc": "V1", "node1": 2, "node2": 4, "value": E},
        {"desc": "R01", "node1": 4, "node2": 1, "value": R0},
        {"desc": "I", "node1": 2, "node2": 0, "value": 0.0},
    ])
    Vn, Ib, It, Vt, Pt = solve_circuit(nl, current=2.0)
    assert abs(Vt[0] - (2 * E - 2.0 * (2 * R0))) < 1e-9


# ---------- 功率控制(缺陷2回归) ----------
def test_power_control_feasible():
    """可行功率：V·I = P，且取高电压物理根。"""
    E, R0 = 3.9, 0.05  # Pmax = E²/4R0 ≈ 76W
    nl = _nl([
        {"desc": "V0", "node1": 1, "node2": 2, "value": E},
        {"desc": "R00", "node1": 2, "node2": 0, "value": R0},
        {"desc": "I", "node1": 1, "node2": 0, "value": 0.0},
    ])
    Vn, Ib, It, Vt, Pt = solve_circuit(nl, power=50.0)
    assert abs(Vt[0] * It[0] - 50.0) < 1e-6
    assert Vt[0] > E / 2.0, "应取高电压(低电流)物理根"


def test_power_control_infeasible_raises():
    """超出 Pmax 的功率需求应抛 ValueError。"""
    E, R0 = 3.9, 0.05  # Pmax ≈ 76W
    nl = _nl([
        {"desc": "V0", "node1": 1, "node2": 2, "value": E},
        {"desc": "R00", "node1": 2, "node2": 0, "value": R0},
        {"desc": "I", "node1": 1, "node2": 0, "value": 0.0},
    ])
    with pytest.raises(ValueError, match="最大可输出功率"):
        solve_circuit(nl, power=200.0)


# ---------- R>0 校验(缺陷3回归) ----------
def test_zero_resistance_rejected():
    """R=0 应在校验阶段抛 ValueError，不得静默产生 inf/nan。"""
    nl = _nl([
        {"desc": "V0", "node1": 1, "node2": 2, "value": 3.0},
        {"desc": "R00", "node1": 2, "node2": 0, "value": 0.0},
        {"desc": "I", "node1": 1, "node2": 0, "value": 1.0},
    ])
    with pytest.raises(ValueError, match="电阻必须为正"):
        solve_circuit(nl, current=1.0)


def test_negative_resistance_rejected():
    nl = _nl([
        {"desc": "V0", "node1": 1, "node2": 2, "value": 3.0},
        {"desc": "R00", "node1": 2, "node2": 0, "value": -0.01},
        {"desc": "I", "node1": 1, "node2": 0, "value": 1.0},
    ])
    with pytest.raises(ValueError, match="电阻必须为正"):
        solve_circuit(nl, current=1.0)


# ---------- setup_circuit 节点完整性 ----------
def test_setup_circuit_cell_count():
    """setup_circuit(nS,nP) 应生成 nS*nP 个 V 元素与 R0 元素。"""
    nl, v_rows, ri_rows = setup_circuit(3, 2)
    assert len(v_rows) == 6
    assert len(ri_rows) == 6
    assert nl.n_cells == 6


# ---------- setup_two_group active 对齐(缺陷回归) ----------
def test_two_group_active_aligned_with_v_order():
    """active_par 必须与网表 V 元素顺序一致（每级先 A 后 B）。"""
    nS = 4
    nl_solo, act_solo, nl_par, act_par = setup_two_group(nS)
    df = nl_par.df
    v_descs = df[df["desc"].str[0] == "V"]["desc"].tolist()
    # V 元素顺序应为 V0, V4, V1, V5, V2, V6, V3, V7
    expected_v = [f"V{s}" for s in range(nS) for _ in (0, 1)]
    expected_v = [f"V{s}" if i == 0 else f"V{nS + s}"
                  for s in range(nS) for i in (0, 1)]
    assert v_descs == expected_v
    # active 与 V 元素电芯编号一致
    expected_active = [s if i == 0 else nS + s for s in range(nS) for i in (0, 1)]
    assert act_par == expected_active


# ---------- setup_series_bypass ----------
def test_series_bypass_removes_faulty_cell():
    """旁路后 V 元素数 = n-1，被旁路芯不在 active 中。"""
    nl, active = setup_series_bypass(8, bypass_idx=3)
    assert nl.n_cells == 7
    assert 3 not in active
    assert len(active) == 7

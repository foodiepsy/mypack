# 整包耦合与拓扑热切换单元测试
import numpy as np

import ecm_pack as ep


def _spec(soc_init=1.0, capacity=5.0):
    return ep.ECMCellSpec(
        capacity=capacity, ocv=lambda s: 3.2 + 0.95 * s,
        R0=0.01, R=[0.005], C=[6000.0], dUdT=-1e-4, soc_init=soc_init,
    )


def test_pack_current_control_single_string():
    """2S 单串恒流放电：端口电压 ≈ 2·单芯电压 - I·ΣR0，SoC 下降。"""
    specs = [_spec(0.9) for _ in range(2)]
    cells = [ep.ECMCell(sp) for sp in specs]
    nl, _, _ = ep.setup_circuit(2, 1)
    pack = ep.Pack(cells, nl, v_cut_lower=2.5)
    out = pack.solve(dt=10.0, control=2.0, control_type="current", n_steps=10)
    # 端口电压为正且随时间下降
    Vt = out["Pack terminal voltage [V]"]
    assert np.all(Vt > 0)
    assert Vt[-1] < Vt[0]
    # SoC 下降
    assert out["Cell SoC"][-1, 0] < out["Cell SoC"][0, 0]


def test_topology_switch_preserves_cell_state():
    """拓扑切换不改变断开电芯的 SoC（状态冻结）。"""
    nS = 4
    specs = [_spec(0.8) for _ in range(nS)] + [_spec(1.0) for _ in range(nS)]
    cells = [ep.ECMCell(sp) for sp in specs]
    nl_solo, act_solo, nl_par, act_par = ep.setup_two_group(nS)
    pack = ep.Pack(cells, nl_solo, active=act_solo, v_cut_lower=2.0)
    # solo 跑 60 步
    pack.solve(dt=1.0, control=2.0, control_type="current", n_steps=60, record_every=60)
    soc_A_before = cells[0].soc
    soc_B_before = cells[nS].soc  # 待机组
    # 切换到 par
    pack.set_topology(nl_par, act_par)
    # 切换本身不改变任何电芯 SoC
    assert abs(cells[0].soc - soc_A_before) < 1e-12
    assert abs(cells[nS].soc - soc_B_before) < 1e-12


def test_topology_switch_produces_circulation():
    """8S→8S2P 切换瞬间应出现环流：高压组放电、低压组被充电。"""
    nS = 4
    specs = [_spec(0.5) for _ in range(nS)] + [_spec(1.0) for _ in range(nS)]
    cells = [ep.ECMCell(sp) for sp in specs]
    nl_solo, act_solo, nl_par, act_par = ep.setup_two_group(nS)
    pack = ep.Pack(cells, nl_solo, active=act_solo, v_cut_lower=2.0)
    # solo 跑一会拉低 A 组
    pack.solve(dt=1.0, control=1.0, control_type="current", n_steps=10, record_every=10)
    # 切换后解一次电路
    pack.set_topology(nl_par, act_par)
    from ecm_pack.circuit import solve_circuit
    Vn, Ib, It, Vt, Pt = solve_circuit(pack.netlist, current=0.0)
    cA = -Ib[0]   # group A cell0
    cB = -Ib[1]   # group B cell(nS)（V 元素顺序第二位）
    # 高压 B 组放电、低压 A 组被充电
    assert cB > 0, "高压组应放电(环流)"
    assert cA < 0, "低压组应被充电(环流)"


def test_recorded_output_shapes():
    """输出数组维度与 N_total 对齐，未接入电芯电流记 0。"""
    nS = 4
    specs = [_spec(0.9) for _ in range(nS)] + [_spec(1.0) for _ in range(nS)]
    cells = [ep.ECMCell(sp) for sp in specs]
    nl_solo, act_solo, _, _ = ep.setup_two_group(nS)
    pack = ep.Pack(cells, nl_solo, active=act_solo, v_cut_lower=2.0)
    out = pack.solve(dt=1.0, control=1.0, control_type="current", n_steps=5, record_every=1)
    N = 2 * nS
    assert out["Cell current [A]"].shape == (6, N)  # 5 步 + t=0
    # 未接入的 group B 电流应全部为 0
    assert np.all(out["Cell current [A]"][:, nS:] == 0.0)

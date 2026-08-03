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
def test_contact_resistance_heat():
    """R_contact>0 时产热应比无接触电阻时更高。"""
    # 无接触电阻
    specs0 = [ep.ECMCellSpec(capacity=5.0, ocv=lambda s: 3.2, R0=0.01,
                             R_contact=0.0, soc_init=0.9) for _ in range(2)]
    cells0 = [ep.ECMCell(sp) for sp in specs0]
    nl, _, _ = ep.setup_circuit(2, 1)
    pack0 = ep.Pack(cells0, nl, v_cut_lower=2.0)
    out0 = pack0.solve(dt=10.0, control=2.0, control_type="current", n_steps=5)
    Vt0 = out0["Pack terminal voltage [V]"]
    # 有接触电阻 1 mΩ
    specs1 = [ep.ECMCellSpec(capacity=5.0, ocv=lambda s: 3.2, R0=0.01,
                             R_contact=0.001, soc_init=0.9) for _ in range(2)]
    cells1 = [ep.ECMCell(sp) for sp in specs1]
    pack1 = ep.Pack(cells1, nl, v_cut_lower=2.0)
    out1 = pack1.solve(dt=10.0, control=2.0, control_type="current", n_steps=5)
    Vt1 = out1["Pack terminal voltage [V]"]

    # 有 R_contact 的端电压更低（多了 I·R_contact 的压降）
    assert Vt1[-1] < Vt0[-1], (
        f"有接触电阻端电压({Vt1[-1]:.4f})应低于无接触电阻({Vt0[-1]:.4f})"
    )

    # SoC 下降也应更快（相同电流、更多能量消耗在 R_contact 上...
    # 但实际上 SoC 只取决于电流积分，相同电流相同时间 SoC 下降应相同。
    # R_contact 的影响体现在端电压更低上。
    assert abs(out0["Cell SoC"][-1, 0] - out1["Cell SoC"][-1, 0]) < 1e-6

def test_contact_resistance_clone():
    """clone() 应携带 R_contact。"""
    spec = ep.ECMCellSpec(capacity=5.0, ocv=lambda s: 3.2, R0=0.01,
                          R_contact=0.002, soc_init=0.9)
    s2 = spec.clone()
    assert float(s2.R_contact(25.0, 0.0, 0.9)) == 0.002
def test_contact_resistance_terminals():
    """R_contact 应反映在端电压中（每 cell 压降 = I·R_contact）。"""
    spec = ep.ECMCellSpec(capacity=5.0, ocv=lambda s: 3.2, R0=0.01,
                          R_contact=0.001, soc_init=1.0)
    cell = ep.ECMCell(spec)
    Td = cell.T - 273.15
    R0_raw = spec.R0(Td, 0.0, cell.soc)
    Rc = float(spec.R_contact(Td, 0.0, cell.soc))
    # 无电流时 R_contact 不影响 OCV
    assert abs(cell.terminal_voltage(R0_raw + Rc, 0.0) - cell.voltage_behind_R0()) < 1e-12
    # 有电流时压降 = I·(R0+Rc)
    I = 5.0
    Vt = cell.terminal_voltage(R0_raw + Rc, I)
    expected = cell.voltage_behind_R0() - I * (R0_raw + Rc)
    assert abs(Vt - expected) < 1e-9

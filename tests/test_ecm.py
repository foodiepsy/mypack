# 电芯 ECM 层单元测试
import numpy as np

import ecm_pack as ep


def _spec(**kw):
    base = dict(
        capacity=1.0, ocv=lambda s: 3.0 + s, R0=0.01,
        R=[0.005], C=[6000.0], soc_init=0.8,
    )
    base.update(kw)
    return ep.ECMCellSpec(**base)


def test_soc_drop_linear():
    """恒流放电 SoC 线性下降：ΔSoC = I·t/(Q·3600)。"""
    cell = ep.ECMCell(_spec())
    I, dt, steps = 2.0, 1.0, 300
    for _ in range(steps):
        cell.step_electrical(I, dt)
    expect = 0.8 - I * dt * steps / (1.0 * 3600.0)
    assert abs(cell.soc - expect) < 1e-9


def test_rc_converges_to_steady():
    """RC 支路解析积分：恒流足够长时间后 v_rc -> -I·R（稳态）。"""
    R1 = 0.005
    cell = ep.ECMCell(_spec(R=[R1], C=[6000.0]))  # tau = R*C = 30s
    I = 1.0
    for _ in range(2000):  # 远超 5*tau
        cell.step_electrical(I, 1.0)
    assert abs(cell.v_rc[0] - (-I * R1)) < 1e-3


def test_diffusion_mass_conservation():
    """回归(缺陷1)：扩散分布 SoC 均值应始终守恒到 bulk SoC（Neumann 边界 -2r）。"""
    cell = ep.ECMCell(_spec(diffusion=True, tau_D=50.0, nx=12))
    for _ in range(400):
        cell.step_electrical(1.0, 1.0)
    assert abs(cell.z.mean() - cell.soc) < 1e-3


def test_terminal_voltage_drops_on_discharge():
    """放电时端电压低于开路电压（E - I·R0）。"""
    cell = ep.ECMCell(_spec())
    E = cell.voltage_behind_R0()
    R0 = cell.spec.R0(25.0, 1.0, cell.soc)
    Vt = cell.terminal_voltage(R0, 1.0)
    assert Vt < E
    assert abs((E - Vt) - 1.0 * R0) < 1e-9

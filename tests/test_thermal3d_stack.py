# StackThermal3D（多电芯三维复合热模型）单元测试
import numpy as np
import pytest

import ecm_pack as ep
from ecm_pack.thermal3d_stack import StackThermal3D

Lx, Ly, Lz = 0.174, 0.0717, 0.207
K = (12.0, 0.7, 11.6)
RHO, CP = 2300.0, 1000.0
TAMB = 298.15


def _residual(st, Q):
    """离散残差 (A·T - C·T)/dt - (src + bc_rhs)，稳态应≈0（能量平衡）。"""
    Q = np.asarray(Q, dtype=float)
    src = np.zeros(st.N)
    for n in range(st.N):
        c = st._cell[n]
        if c >= 0:
            src[n] = Q[c] * st._vol[n] / st._V_cell
    return (st._A @ st.T_field - st._C * st.T_field) / st._dt - (src + st._bc_rhs)


def test_energy_balance_single_cell():
    """单芯(薄侧泡棉+空气)冷却：稳态离散残差相对产热应很小（能量平衡）。"""
    Q = 50.0
    st = StackThermal3D(1, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                        foam_k=0.04, foam_thickness=0.001,
                        h_top=20, T_top=TAMB, h_bottom=20, T_bottom=TAMB,
                        h_side=20, T_amb=TAMB, T_init=TAMB)
    for _ in range(4000):
        st.step(np.array([Q]), dt=10.0)
    r = _residual(st, np.array([Q]))
    assert abs(r.sum()) < 0.02 * Q, f"能量不平衡: 残差={r.sum():.3f}W"


def test_foam_side_insulation_direction():
    """薄侧泡棉越差(低k)，侧面散热越弱、整芯越热（方向正确）。
    仅用薄侧散热(顶/底绝热)以隔离泡棉影响，得到清晰的方向差。"""
    Q2 = np.array([60.0, 20.0])
    kw = dict(h_top=0, T_top=TAMB, h_bottom=0, T_bottom=TAMB,
               h_side=50, T_amb=TAMB, T_init=TAMB)
    # 差泡棉（薄侧为关键热阻）
    s_bad = StackThermal3D(2, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                           foam_k=0.005, foam_thickness=0.001, **kw)
    for _ in range(5000):
        s_bad.step(Q2, dt=20.0)
    # 好泡棉
    s_good = StackThermal3D(2, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                            foam_k=5.0, foam_thickness=0.001, **kw)
    for _ in range(5000):
        s_good.step(Q2, dt=20.0)
    t_bad = s_bad.T.mean()
    t_good = s_good.T.mean()
    assert t_bad > t_good, f"薄侧泡棉方向错误: 差{t_bad:.1f}K 好{t_good:.1f}K"
    # 能量平衡（仅侧面散热、弱导热使平衡较慢，留 5% 余量）
    assert abs(_residual(s_bad, Q2).sum()) < 0.05 * Q2.sum()
    assert abs(_residual(s_good, Q2).sum()) < 0.05 * Q2.sum()


def test_per_cell_temperature_length():
    """StackThermal3D.T 长度须等于电芯数，供 Pack 接入；
    且按用户修正：电芯之间(沿Y)不应有泡棉块，泡棉只在薄侧。"""
    st = StackThermal3D(8, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                        foam_k=0.04, foam_thickness=0.001,
                        h_top=50, T_top=TAMB, h_bottom=0, T_bottom=TAMB,
                        h_side=50, T_amb=TAMB, T_init=TAMB)
    assert st.T.shape == (8,)
    # 关键修正：沿 Y 的每个切片都是电芯，不含任何泡棉块
    assert not any(p.get("foam", False) for p in st._per_j), \
        "电芯之间不应有泡棉块（泡棉只在薄侧）"
    # 底部绝热、顶部冷板：bat8 应不比 bat1 凉（非对称生效）
    for _ in range(300):
        st.step(np.full(8, 60.0), dt=10.0)
    assert st.T[7] >= st.T[0] - 1e-6

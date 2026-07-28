# StackThermal3D unit tests
import numpy as np
import pytest

from ecm_pack.thermal3d_stack import StackThermal3D

Lx, Ly, Lz = 0.174, 0.0717, 0.207
K = (12.0, 0.7, 11.6)
RHO, CP = 2300.0, 1000.0
TAMB = 298.15


def _residual(st, Q):
    Q = np.asarray(Q, dtype=float)
    src = np.zeros(st.N)
    for n in range(st.N):
        c = st._cell[n]
        if c >= 0:
            src[n] = Q[c] * st._vol[n] / st._V_cell
    return (st._A @ st.T_field - st._C * st.T_field) / st._dt - (src + st._bc_rhs)


def test_energy_balance_single_cell():
    Q = 50.0
    st = StackThermal3D(1, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                        foam_k=0.04, foam_thickness=0.001,
                        h_top=20, T_top=TAMB, h_bottom=20, T_bottom=TAMB,
                        h_side=20, T_amb=TAMB, T_init=TAMB)
    for _ in range(4000):
        st.step(np.array([Q]), dt=10.0)
    r = _residual(st, np.array([Q]))
    assert abs(r.sum()) < 0.02 * Q, f"energy imbalance: {r.sum():.3f}W"


def test_foam_side_insulation_direction():
    Q2 = np.array([60.0, 20.0])
    kw = dict(h_top=0, T_top=TAMB, h_bottom=0, T_bottom=TAMB,
               h_side=50, T_amb=TAMB, T_init=TAMB)
    s_bad = StackThermal3D(2, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                           foam_k=0.005, foam_thickness=0.001, **kw)
    for _ in range(5000):
        s_bad.step(Q2, dt=20.0)
    s_good = StackThermal3D(2, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                            foam_k=5.0, foam_thickness=0.001, **kw)
    for _ in range(5000):
        s_good.step(Q2, dt=20.0)
    t_bad = s_bad.T.mean()
    t_good = s_good.T.mean()
    assert t_bad > t_good, f"foam direction wrong: bad={t_bad:.1f} good={t_good:.1f}"
    assert abs(_residual(s_bad, Q2).sum()) < 0.05 * Q2.sum()
    assert abs(_residual(s_good, Q2).sum()) < 0.05 * Q2.sum()


def test_per_cell_temperature_length():
    st = StackThermal3D(8, Lx, Ly, Lz, 2, 3, 3, K, RHO, CP,
                        foam_k=0.04, foam_thickness=0.001,
                        foam_faces=["x0", "z0"],
                        k_top=0.2, d_top=2e-4,
                        h_top=50, T_top=TAMB, h_bottom=0, T_bottom=TAMB,
                        h_side=50, T_amb=TAMB, T_init=TAMB)
    assert st.T.shape == (8,)
    # No foam blocks between cells (only on thin sides)
    assert not any(p.get("foam", False) for p in st._per_j)
    # Asymmetric thin sides configured
    assert st.foam_faces == {"x0", "z0"}
    for _ in range(300):
        st.step(np.full(8, 60.0), dt=10.0)
    assert st.T[7] >= st.T[0] - 1e-6

"""核心数值内核的直接单元测试（回归锁）。
聚焦 ecm_pack 中"易错且影响正确性"的数值核心：
Thomas 三对角求解器
导热矩阵对角化（对角 = -行非对角和，保证 Σ_j K_ij T_j = Σ g(T_j-T_i)）
分布 SoC 扩散的 -2r 边界（虚节点法，保证质量守恒）
电路 MNA 功率控制闭式解
堆叠热模型 Y 向谐波平均电导
StackThermal3D.step 热源分配/均值矩阵化（与逐节点循环等价）
"""
import numpy as np
import pytest
import ecm_pack as ep
from ecm_pack.ecm import _thomas
from ecm_pack.thermal3d_stack import StackThermal3D
def test_thomas_matches_dense():
    rng = np.random.default_rng(42)
    for n in range(2, 10):
        main = rng.uniform(1.0, 3.0, n)
        lower = rng.uniform(-1.0, -0.1, n - 1)
        upper = rng.uniform(-1.0, -0.1, n - 1)
        rhs = rng.uniform(-1.0, 1.0, n)
        A = np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)
        x = _thomas(main, lower, upper, rhs)
        assert np.allclose(A @ x, rhs, atol=1e-9)
    # 手算 3x3
    main = np.array([2.0, 3.0, 2.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([-1.0, -1.0])
    rhs = np.array([1.0, 2.0, 1.0])
    A = np.diag(main) + np.diag(lower, -1) + np.diag(upper, 1)
    x = _thomas(main, lower, upper, rhs)
    assert np.allclose(x, np.linalg.solve(A, rhs), atol=1e-12)

def test_build_conduction_diagonal_is_neg_rowsum():
    tn = ep.ThermalNetwork(
    n_cells=5, C_th=100.0,
    conduction=[(0, 1, 2.0), (1, 2, 3.0), (3, 4, 1.5)],
    )
    K = tn.K
    # 对称
    assert np.allclose(K, K.T, atol=1e-12)
    # 每行和为 0（对角 = -行非对角和）
    assert np.allclose(K.sum(axis=1), 0.0, atol=1e-12)
    # 无导热 -> 零矩阵
    tn2 = ep.ThermalNetwork(n_cells=4, C_th=100.0, conduction=None)
    assert np.allclose(tn2.K, 0.0)

def _make_diffusion_cell(soc=0.5):
    spec = ep.cell_314ah_spec(soc_init=soc)
    spec.diffusion = True
    spec.tau_D = 50.0
    spec.nx = 8
    spec.capacity = 314.0
    return ep.ECMCell(spec)
    def test_diffusion_2r_boundary_mass_conservation():
        cell = _make_diffusion_cell(soc=0.5)
        cell.reset()
        assert cell.z is not None
    # 不均匀初始：总和守恒（I=0，J=0）
    cell.z = 0.3 + 0.05 * np.linspace(-1.0, 1.0, cell.spec.nx)
    z0_sum = cell.z.sum()
    cell._step_diffusion(0.0, dt=30.0)
    assert np.allclose(cell.z.sum(), z0_sum, atol=1e-10)

    # 均匀初始保持均匀（I=0）
    cell.z = np.full(cell.spec.nx, 0.7)
    cell._step_diffusion(0.0, dt=100.0)
    assert np.allclose(cell.z, 0.7, atol=1e-12)

def test_power_control_closed_form():
    netlist, v_rows, ri_rows = ep.setup_circuit(8, 1, Rbus=0.0)
    E = np.linspace(3.30, 3.35, 8)
    R0 = np.linspace(0.0004, 0.0006, 8)
    for k, vr in enumerate(v_rows):
        netlist.df.at[vr, "value"] = E[k]
        for k, rr in enumerate(ri_rows):
            netlist.df.at[rr, "value"] = R0[k]
    P_req = 1500.0
    _Vn, _Ib, I_term, V_term, P_term = ep.solve_circuit(netlist, power=P_req)
    I = float(np.asarray(I_term).reshape(-1)[0])
    Vt = float(np.asarray(V_term).reshape(-1)[0])
    P_actual = float(np.asarray(P_term).reshape(-1)[0])
    # P ≈ V_terminal · I
    assert abs(P_actual - P_req) < 1e-6
    assert abs(P_actual - Vt * I) < 1e-6

    # 不可行功率抛 ValueError
    V_oc = E.sum()
    R_eq = R0.sum()
    Pmax = V_oc ** 2 / (4.0 * R_eq)
    with pytest.raises(ValueError):
        ep.solve_circuit(netlist, power=Pmax * 1.1)

def test_gy_conductance_harmonic_mean():
    st = StackThermal3D(
    n_cells=2,
    Lx=0.174, Ly=0.0717, Lz=0.207,
    nx=3, ny=3, nz=3,
    cell_k=(12.0, 0.7, 11.6),
    cell_rho=2300.0, cell_cp=1000.0,
    foam_k=0.05, foam_thickness=0.001,
    )
    # 取同芯内两相邻 Y 层
    n = st._idx(1, 1, 1)
    nb = st._idx(1, 2, 1)
    Ay = st.dx * st.dz
    g = st._gy_conductance(n, nb, Ay)
    kA = st._ky[n]; dA = st._dy[n]
    kB = st._ky[nb]; dB = st._dy[nb]
    expected = Ay / ((dA / 2.0) / kA + (dB / 2.0) / kB)
    assert abs(g - expected) < 1e-12
    def test_stack3d_step_matrix_equivalence():
        """step 矩阵化（S@Q / W@T）须与逐节点循环、逐芯 mask.mean 等价。"""
        spec = ep.cell_314ah_spec()
        st = StackThermal3D(
            n_cells=8, Lx=spec.Lx, Ly=spec.Ly, Lz=spec.Lz,
            nx=4, ny=5, nz=8,
            cell_k=spec.k, cell_rho=spec.rho, cell_cp=spec.cp,
            foam_k=0.04, foam_thickness=0.001, foam_faces=["x0", "z0"],
            k_top=0.2, d_top=0.0002, h_top=50, h_bottom=50, h_side=50, T_init=298.15,
        )
        # W 行和须为 1（均值归一化）
        assert np.allclose(np.asarray(st._W.sum(axis=1)).ravel(), 1.0, atol=1e-12)
        Q = np.linspace(1.0, 8.0, 8)
        src_loop = np.zeros(st.N)
        for nn in range(st.N):
            c = st._cell[nn]
            if c >= 0:
                src_loop[nn] = Q[c] * st._vol[nn] / st._V_cell
        assert np.max(np.abs((st._S @ Q) - src_loop)) < 1e-12

        Tf = np.random.default_rng(0).uniform(298.0, 303.0, st.N)
        T_loop = np.array([float(Tf[st._cell == c].mean()) for c in range(st.n_cells)])
        assert np.max(np.abs(np.asarray(st._W @ Tf).ravel() - T_loop)) < 1e-10

    test_stack3d_step_matrix_equivalence()

def test_r0_r1_soc_t_table_matches_analytic():
    """R0-SOC-T 查表钩子：启用查表后应与解析式一致，且电流项仍叠加。
    必须在 finally 还原全局钩子，避免污染其它测试。"""
    from ecm_pack import defaults
    pts = [(s, T) for s in (0.1, 0.3, 0.5, 0.8, 0.95)
           for T in (-10.0, 0.0, 25.0, 45.0, 55.0)]
    # 1) 解析式路径（钩子未启用）下记录基准值
    r0_ana = {pt: ep.R0_314ah(pt[1], 0.0, pt[0]) for pt in pts}
    r1_ana = {pt: ep.R1_314ah(pt[1], 0.0, pt[0]) for pt in pts}
    try:
        # 2) 由解析式预生成等价 (SoC, T) 查表并启用
        defaults.R0_SOC_T_TABLE = defaults.build_r0_soc_t_table()
        defaults.R1_SOC_T_TABLE = defaults.build_r1_soc_t_table()
        for s, T in pts:
            assert abs(ep.R0_314ah(T, 0.0, s) - r0_ana[(s, T)]) < 1e-6
            assert abs(ep.R1_314ah(T, 0.0, s) - r1_ana[(s, T)]) < 1e-6
        # 3) 电流项仍叠加（与解析式路径的 I 项一致）
        base = ep.R0_314ah(25.0, 0.0, 0.5)
        assert abs(ep.R0_314ah(25.0, 100.0, 0.5) - (base + 1e-7 * 100.0)) < 1e-12
    finally:
        defaults.R0_SOC_T_TABLE = None
        defaults.R1_SOC_T_TABLE = None

def _fill_and_solve(nl, E, I_load):
    """给网表所有 V 元素赋同一电动势后求解，返回 (Vt, 每支路电流)。"""
    df = nl.df
    for i in np.where(df["desc"].str[0] == "V")[0]:
        df.at[i, "value"] = E
        _, Ib, _, Vt, _ = ep.solve_circuit(nl, current=I_load)
        return float(np.ravel(Vt)[0]), np.ravel(Ib)
        @pytest.mark.parametrize("Rbus", [0.0, 1e-4, 1e-3, 1e-2])
        def test_setup_circuit_rbus_matches_analytic(Rbus):
            """母排电阻 Rbus 必须真正串进回路。
    回归 bug：旧实现算出了母排节点 `left` 却没把电芯 R0 接上去（节点悬空），
    且母排节点号与下一串级的首个私有节点撞号，导致 Rbus>0 时电路病态——
    4S2P/100A 下每支路电流曾算出 -735A（正确值 -50A）。
    """
    nS, nP, E, R0, I_load = 4, 2, 3.3, 1e-3, 100.0
    nl, _, _ = ep.setup_circuit(nS, nP, Rbus=Rbus)
    vt, ib = _fill_and_solve(nl, E, I_load)

    # 解析解：每串级等效阻 = R0/nP + Rbus；Vt = nS*E - I*nS*(R0/nP + Rbus)
    ana = nS * E - I_load * nS * (R0 / nP + Rbus)
    assert abs(vt - ana) < 1e-9, f"Rbus={Rbus}: Vt={vt} 解析={ana}"
    # 并联支路均分总电流，且与 Rbus 无关
    assert np.allclose(np.abs(ib), I_load / nP, atol=1e-9)

def test_setup_circuit_rbus_zero_is_unchanged():
    """Rbus=0 时网表必须与修复前逐元素完全一致（保证零回归）。"""
    nl0, _, _ = ep.setup_circuit(8, 1, Rbus=0.0)
    df = nl0.df
    # 无母排元素，且所有 R0 的 node2 就是本串级负端 s
    assert not (df["desc"].str[:2] == "Rb").any()
    r0 = df[df["desc"].str[:2] == "R0"].reset_index(drop=True)
    assert list(r0["node2"]) == list(range(8))
    def test_setup_two_group_rejects_unimplemented_rbus():
        """Rbus 曾被 setup_two_group 静默忽略，现在必须显式报错而非算错。"""
        ep.setup_two_group(4, Rbus=0.0)  # 允许
        with pytest.raises(NotImplementedError):
            ep.setup_two_group(4, Rbus=1e-3)
            def test_pack_rint_equals_r0_not_double():
                """Pack 输出的 "Cell internal resistance" 必须等于真实 R0，而不是 2×R0。
    回归 bug：曾写作 abs((E + I*R0 - Vt)/I)，因 Vt = E - I*R0，
    分子展开为 2*I*R0，导致输出内阻整整放大一倍。
    """
    n = 4
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=0.8, T_init=298.15)) for _ in range(n)]
    nl, _, _ = ep.setup_circuit(n, 1)
    pack = ep.Pack(cells, nl, v_cut_lower=2.0)
    I_load = 157.0
    out = pack.solve(dt=2.0, control=I_load, control_type="current",
                     n_steps=10, record_every=1)

    rint = out["Cell internal resistance [Ohm]"][-1]
    # 与独立推进的参考电芯对比真实 R0（同一 SoC/T/I 轨迹）
    ref = ep.ECMCell(ep.cell_314ah_spec(soc_init=0.8, T_init=298.15))
    for _ in range(10):
        r0_ref = ref.step_electrical(I_load, 2.0)

    assert np.allclose(rint, r0_ref, rtol=1e-6), (
        f"Rint={rint[0]:.8f} 与真实 R0={r0_ref:.8f} 不符 "
        f"(比值 {rint[0] / r0_ref:.4f}，若≈2 则是历史的 2× bug)"
    )
    # 量级理性检查：314Ah 电芯 R0 是 0.4mΩ 量级，绝不该到 mΩ 级
    assert np.all(rint < 1e-3), f"Rint 量级异常: {rint}"

def test_pack_strict_raises_on_bad_initial_solve():
    """strict=True（默认）时，初始电路无解必须抛错，而不是给一个零值快照。"""
    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=0.5)) for _ in range(2)]
    nl, _, _ = ep.setup_circuit(2, 1)
    # 功率控制下要一个物理上不可能达到的功率 -> 闭式解无实根
    pack = ep.Pack(cells, nl)
    with pytest.raises(RuntimeError, match="初始电路求解失败"):
        pack.solve(dt=1.0, control=1e12, control_type="power", n_steps=2)
        def test_pack_strict_false_falls_back():
            """strict=False 时保留历史兜底行为（零值快照 + error 日志），不抛错。"""
            cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=0.5)) for _ in range(2)]
            nl, _, _ = ep.setup_circuit(2, 1)
            pack = ep.Pack(cells, nl, strict=False)
            # 初始解兜底成功后，主循环仍会因同样的不可行功率而终止 —— 这符合预期，
            # 这里只断言"初始解这一步"不再抛 '初始电路求解失败'。
            with pytest.raises(RuntimeError) as ei:
                pack.solve(dt=1.0, control=1e12, control_type="power", n_steps=2)
                assert "初始电路求解失败" not in str(ei.value)
                def test_pack_solve_error_message_has_step_context():
                    """主循环求解失败必须带上时间/步号，而不是裸抛 LinAlgError。"""
                    cells = [ep.ECMCell(ep.cell_314ah_spec(soc_init=0.5)) for _ in range(2)]
                    nl, _, _ = ep.setup_circuit(2, 1)
                    pack = ep.Pack(cells, nl, strict=False)
                    with pytest.raises(RuntimeError, match=r"step \d+/\d+"):
                        pack.solve(dt=1.0, control=1e12, control_type="power", n_steps=2)

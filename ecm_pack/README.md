# ecm_pack —�? 可定�? ECM + 电路耦合 + 热网�? 的电池包仿真�?

一�?**自包含、零重型依赖**（仅 numpy / scipy / pandas / matplotlib）的电池包仿真库�?
设计范式直接源自以下两个开源库�?

- **PyBaMM �? ECM**（`pybamm.equivalent_circuit.Thevenin`）：把电芯建模为
  `OCV + R0 + Σ(RC) [+ 扩散过电势]` 的可插拔等效电路�?
- **liionpack** 的耦合方式：把每个电芯�?**网表**里表示为「电压源 + 串联电阻」，
  �?**修正节点�?(MNA)**解出每支路电流，再回灌给电化学模型做一步推进（“双步循环”）�?

本库�?**可定制的 ECM 电芯**替换 liionpack 里的黑盒 PyBaMM 模型，并补齐�?
**热模�?**�?**电芯间自定义导热**�?**拓扑热切换（可重构电池包�?**，从而一次性满足以下工程需求：

1. **ECM 模型定制** —�? RC 阶数可调，OCV/R0/Rk/Ck/dUdT 支持�? (T, I, SoC) 的查表与解析式，可�? ECMD 扩散过电�?
2. **自定义串并联拓扑** —�? 标准 `nS nP` 生成�? + 任意手写网表（MNA 修正节点法求解，支持电流/功率控制�?
3. **热模型集�?** —�? 集总热网络 + **三维有限体积热模�?**，支�? 1D / 2D / 3D 自由切换�?
   各向异性导热系�? (kx, ky, kz)、比热容、密度均可自定义
4. **自定义电芯间导热** —�? 任意导热矩阵 / 邻接表，隐式欧拉无条件稳�?
5. **拓扑热切换（可重构电池包�?** —�? 运行中随�?/定时改变整包拓扑�?
   一组工作一段时间后另一组并入并联、故障芯自动旁路�?
6. **工业 314Ah 大电芯默认参�?** —�? 开箱即用的储能电芯规格，含精确几何尺寸与热物性参�?

---

## 安装与运�?

```bash
cd /workspace
# 6 �? demo，按 demo1~demo6 分文件夹，产物在各自�? result/ 子目�?
python3.11 ecm_pack/examples/demo1-8s-foam/demo_8s_foam_thermal.py    # 8S 大面背靠�? + 薄侧泡棉 + 三维热模�?
python3.11 ecm_pack/examples/demo2-8s2p/demo_8s2p_200a.py             # 8S2P / 200A 环流工况
python3.11 ecm_pack/examples/demo3-314ah-3d/demo_314ah_3d.py          # 314Ah 大电�? 1D/2D/3D 热模型对�?
python3.11 ecm_pack/examples/demo4-basic/demo.py                       # ECM + 拓扑 + 热模�? + 导热（四要素�?
python3.11 ecm_pack/examples/demo5-fault-tolerant/fault_tolerant_demo.py  # 故障容错：缺陷芯到阈值自动旁�?
python3.11 ecm_pack/examples/demo6-topology/topology_demo.py           # 拓扑热切换：运行中并入另一�? 8S
```

库本身无需安装，只�? `ecm_pack/` �? `PYTHONPATH` 即可 `import ecm_pack as ep`�?

依赖：`numpy`, `scipy`, `pandas`, `matplotlib`（均 pip 安装的纯 Python 库）�?

---

## 模块结构

| 文件 | 职责 |
|---|---|
| `data.py` | 参数归一化：常量 / 可调�? / 1D·2D·3D 查表统一成可调用函数 |
| `ecm.py` | `ECMCellSpec`（规格）+ `ECMCell`（状态）。RC 支路�?**解析指数积分**推进，无条件稳定；支�? `R_contact` 连接电阻发热 |
| `circuit.py` | `Netlist` + `solve_circuit`（MNA 求解，支持电�?/功率控制�?+ `setup_circuit(nS,nP)` + `setup_two_group` + `setup_series_bypass` |
| `thermal.py` | `ThermalNetwork`：集总热�? + 对流 + 自定义电芯间导热 + **层间热接触电�? R_th**（硅胶垫/气隙界面效应）；隐式欧拉，无条件稳定 |
| `thermal3d.py` | `CellThermalModel`�?**三维有限体积热模�?**，支�? `dim=1,2,3` 切换；各向异�? k=(kx,ky,kz)�?**非对称冷�?**（每面独�? h）；**壳层热阻 R_shell**（T_surface/T_core_max）；内置 `plot_slice()`/`plot_summary()` 可视�? |
| `defaults.py` | **314Ah 大电芯默认参�?**：`cell_314ah_spec()` —�? 精确几何 174×71.7×207mm，热物�? ρ=2300, cp=1000, k=(12,0.7,11.6)，电�? R0=R1=0.4mΩ, τ=100s |
| `pack.py` | `Pack`：双步循环耦合 ECM+电路+热；`set_topology()` �? `solve(topology_events=, switch_callback=)` 支持拓扑热切�? |

---

## 最小用�?

### 方式一：手�? ECM 规格（完全自定义�?

```python
import numpy as np
import ecm_pack as ep

# 1) 定义一�? 1 �? RC �? ECM 电芯（参数可�? (T,I,SoC) 变化�?
spec = ep.ECMCellSpec(
    capacity=5.0,
    ocv=lambda s: 3.2 + 0.95 * s,                 # OCV(SoC)
    R0=lambda T, I, s: 0.01 * (1 + 0.6 * (1 - s)), # R0(T,I,SoC)
    R=[lambda T, I, s: 0.005], C=[lambda T, I, s: 6000.0],
    dUdT=lambda T, s: -1e-4,                         # 可逆热系数
)

# 2) 2 �? 2 并拓扑（也可手写任意网表�?
cells = [ep.ECMCell(spec) for _ in range(4)]
netlist, v_rows, ri_rows = ep.setup_circuit(2, 2)

# 3) 热网络（含自定义导热：芯 0-�? 1 强耦合�?
thermal = ep.ThermalNetwork(4, C_th=300.0, h=0.5, T_amb=298.15,
                            conduction=[(0, 1, 30.0)])

# 4) 耦合求解：整�? 8A 放电 600 步，每步 10s
pack = ep.Pack(cells, netlist, thermal=thermal, v_cut_lower=2.5)
out = pack.solve(dt=10.0, control=8.0, control_type="current", n_steps=600)
print(out["Pack terminal voltage [V]"])
print(out["Cell SoC"])
```

### 方式二：使用 314Ah 大电芯默认参数（开箱即用）

```python
import ecm_pack as ep

# 一行构�? 314Ah 储能大电芯（含完整几何与热物性参数）
spec = ep.cell_314ah_spec(soc_init=1.0, T_init=298.15)
cell = ep.ECMCell(spec)

# 查看内嵌的几�? / 热物性（可用于驱�? 3D 热模型）
print(spec.Lx, spec.Ly, spec.Lz)   # 0.174  0.0717  0.207  [m]
print(spec.rho, spec.cp, spec.k)   # 2300.0  1000.0  (12.0, 0.7, 11.6)
```

---

## 314Ah 大电芯默认参�?

`ep.cell_314ah_spec(soc_init=0.5, T_init=298.15)` 返回一�? **314Ah 储能大电�?** 的完整规格：

| 类别 | 参数 | �? |
|------|------|----|
| **电气** | 容量 | 314 Ah |
| | OCV 曲线 | LFP 体系（平台特�? ~3.2V�? |
| | 欧姆内阻 R0 | **0.4 mΩ** |
| | 极化内阻 R1 | **0.4 mΩ** |
| | RC 时间常数 τ | **100 s**（C1 = τ/R1 = 250 kF�? |
| **几何** | 宽度 X | **174 mm** |
| | 厚度 Y | **71.7 mm** |
| | 高度 Z | **207 mm** |
| **热物�?** | 密度 ρ | **2300 kg/m³** |
| | 比热�? cp | **1000 J/(kg·K)** |
| | 导热系数 k | **(12, 0.7, 11.6) W/(m·K)** —�? 各向异�? |
| | | X(宽度) 12 / Y(厚度) 0.7 / Z(高度) 11.6 |

> 注：厚度方向（Y）导热系数仅 0.7 W/mK，是热管理的瓶颈方向�?

---

## 三维热模型：`CellThermalModel`

支持 **1D / 2D / 3D** 的自由切换，通过 `dim` 参数控制�?

```python
from ecm_pack.thermal3d import CellThermalModel

# 3D 全维度热模型（默认）
tm3 = CellThermalModel(Lx=0.174, Ly=0.0717, Lz=0.207, dim=3,
                       nx=6, ny=6, nz=10,
                       rho=2300.0, cp=1000.0,
                       k=(12.0, 0.7, 11.6),   # 各向异�?
                       h=5.0, T_amb=298.15)

# 1D 模型（仅 X 方向传热，Y/Z �? 1 节点�?
tm1 = CellThermalModel(Lx=0.174, Ly=0.0717, Lz=0.207, dim=1)

# 2D 模型（XY 平面，Z �? 1 节点�?
tm2 = CellThermalModel(Lx=0.174, Ly=0.0717, Lz=0.207, dim=2)

# 推进一步：Q 为产�? [W]，dt 步长 [s]
tm3.step(Q=10.0, dt=60.0, t=0.0)

# 获取温度场分�?
T_field = tm3.reshape()   # dim=1�?(nx,)  dim=2�?(nx,ny)  dim=3�?(nx,ny,nz)
T_mean, T_max, T_min = tm3.stats()
```

**数值方��**：隐式欧拉（后向欧拉），scipy.sparse 稀疏矩阵求解，任意步长 dt 和导热系数均无条件稳定�?

---

## 热模型五大新增功�?

### �? 非对称冷却（面差异化 h�?

```python
# 水冷板在 x0 �? 200 W/m²K，其余自然对�? 5
tm = CellThermalModel(0.174, 0.0717, 0.207, dim=3,
                      h={"x0": 200, "default": 5})
# �? 6-元组: h=(hx0,hx1,hy0,hy1,hz0,hz1)
```

每面独立的对流系数，真实模拟单面水冷 / 非对称散热场景�?

### �? 接触/连接电阻发热

```python
spec = ECMCellSpec(capacity=314, ..., R_contact=0.0005)  # 0.5 mΩ
# Pack 自动叠加: R0_eff = R0 + R_contact
# 产热自动包含: Q = Q_internal + I²·R_contact
```

模拟 tab/焊接/busbar 等效串联电阻的附加压降与发热，高倍率（≥2C）场景不可忽略�?

### �? 层间热接触电�?

```python
thermal = ThermalNetwork(4, C_th=500,
    interface_resistance=[(0, 1, 2e-4, 0.036),   # R_th=2 K·cm²/W, A=0.036 m²
                          (1, 2, 2e-4, 0.036)])
# 自动换算 G = A / R_th，无缝叠加到传导矩阵
```

将硅胶垫/气隙/蓝膜的界面热阻直接作为物理参数输入，G=A/R_th 自动换算�?

### �? 壳层表面温度（BMS 传感器对标）

```python
tm = CellThermalModel(0.174, 0.0717, 0.207, R_shell=0.3)  # 0.3 K/W
tm.step(Q=50, dt=10)
print(tm.T_surface)   # 壳温（对 BMS 传感器读数）
print(tm.T_core_max)  # 体心最高温（内部真实热点）
```

R_shell �? FVM 体温度与环境之间插入一层热阻，分离电芯内部温度与表面可测温度�?

### �? 热场一键可视化

```python
tm.plot_slice("xy")                    # XY 中截�? heatmap
tm.plot_summary(save_path="out.png")   # 三截�? + 统计摘要
```

无需手写 matplotlib 代码，直接出图查看温度分布�?

---

## 拓扑热切换（可重构电池包�?

核心思想�?**电芯的时序状态（SoC/RC/温度）与电路网表完全解�?**。切换拓�? =
替换网表 + 替换「接入映�? `active`」，不触碰任何电芯的 SoC/RC/温度。因此被断开
的电芯状态冻结、新并入的电芯按自身状态立即参与，并联瞬间�? MNA 自动产生环流/浪涌�?

`active` 是「网表第 k �? V 元素 �? cells 中第几个电芯」的映射�?**顺序必须与网�?
V 元素顺序一�?**（见 `examples/demo6-topology/topology_demo.py` �? `active_par` 的踩坑记录）�?

三种切换入口�?

```python
# �? 定时切换：t=600s 时把整包�? 8S 重构�? 8S2P（两�? 8S 并联�?
nl_solo, act_solo, nl_par, act_par = ep.setup_two_group(8)
pack = ep.Pack(cells, nl_solo, active=act_solo)
out = pack.solve(dt=1.0, control=8.0, n_steps=1200,
                 topology_events=[(600.0, nl_par, act_par)])

# �? 事件驱动（随�?/按工况）：返�? (netlist, active) 即切换，返回 None 不切�?
def cb(t, pack):
    if pack.cells[3].terminal_voltage(0.01, 10.0) < 3.4:   # 缺陷芯逼近截止
        return ep.setup_series_bypass(8, bypass_idx=3)      # 自动旁路该芯
    return None
out = pack.solve(dt=1.0, control=10.0, n_steps=800, switch_callback=cb)

# �? 运行中随时手动切�?
pack.set_topology(nl_par, act_par)   # 立即生效，电芯状态零丢失
```

`setup_two_group(nS)` 返回「单�? nS」与「两�? nS 并联(nS2P)」两套网表；
`setup_series_bypass(n, bypass_idx=k)` 返回 n 串联网表，给�? `k` 时旁路第 k 芯�?

---

## 关键设计�?

- **RC 解析积分**：`v_rc(t+dt) = (-I·R) + (v_rc - (-I·R))·exp(-dt/τ)`�?
  对时间步长无条件稳定，无需刚�? ODE 求解器，整包可固定大步长推进�?
- **MNA 耦合**：每个电芯在网表里是「电压源 E_k = OCV+Σv_rc+η_diff」串「电�? R0_k」�?
  �? MNA 得支路电�? I_k，回灌给 ECM 步进 SoC/RC/扩散与产热�?
- **隐式热网�?**：导�? + 对流用后向欧拉求解，任意大导热系�? G 或步长都不会发散�?
- **功率控制**：用 Thevenin 闭式解析�? `I=(V_oc−√(V_oc²�?4R_eq·P))/(2R_eq)`�?
  无需脆弱迭代，并自动�? `P>Pmax` 不可行�?
- **符号约定**：正电流 = 放电（与 PyBaMM 一致）；网表负端接地（node 0），
  电压源正极接电芯正端（b_s），�? `cell_current = -I_batt`�?

---

## 验证结果

**测试覆盖**�?**51 �? pytest** 全部通过（`ecm` / `circuit` / `thermal` / `pack` / `thermal3d`），�? 20 项新增热模型测试�?

**演示�?**（`/workspace/*.png`）：

| 场景 | 结果 |
|---|---|
| 单芯 5A 放电 1h | SoC 0.80 �? 0，电�? 3.95 �? 3.07V �? |
| 2S2P 不均衡放�? | 并联不均流明显（同串级内电流�? 0.08/0.30A）✓ |
| 自定义导�? G=30 | 两芯温度被拉�? 298.625/298.625K �? |
| 功率控制 120W | 端口功率稳定�? 120.0W �? |
| 拓扑热切�? 8S �? 8S2P | 并入瞬间自动产生 ~21.8A 组间环流�?10min �? SoC Δ 0.463�?0.078 �? |
| 故障容错旁路 | 缺陷芯到 3.4V 自动旁路，整包不宕机�?8S�?7S，V 28.9�?25.5V）✓ |
| 314Ah 1D/2D/3D 热对�? | 1D ΔT=0.19K�?2D/3D ΔT~0.74K（厚度方�? k=0.7 W/mK 为瓶颈）�? |
| **8S2P 200A 环流工况** | **并入瞬间环流峰�? 139.7A，稳�? 10.4A；RC 极化失配主导合闸涌流** �? |
| **非对称冷�?** | **强冷面侧温度显著低于对面�?1D 验证 ΔT>0�?** �? |
| **壳层热阻** | **R_shell=0.5 K/W �? T_surface 介于 T_avg �? T_amb 之间** �? |
| **层间热接触电�?** | **R_th×100 使层间温差增�?** �? |

# ecm_pack —— 可定制 ECM + 电路耦合 + 热网络 的电池包仿真库

一个**自包含、零重型依赖**（仅 numpy / scipy / pandas / matplotlib）的电池包仿真库。
设计范式直接源自以下两个开源库：

- **PyBaMM 的 ECM**（`pybamm.equivalent_circuit.Thevenin`）：把电芯建模为
  `OCV + R0 + Σ(RC) [+ 扩散过电势]` 的可插拔等效电路；
- **liionpack** 的耦合方式：把每个电芯在**网表**里表示为「电压源 + 串联电阻」，
  用**修正节点法(MNA)**解出每支路电流，再回灌给电化学模型做一步推进（“双步循环”）。

本库用**可定制的 ECM 电芯**替换 liionpack 里的黑盒 PyBaMM 模型，并补齐了
**热模型**、**电芯间自定义导热**与**拓扑热切换（可重构电池包）**，从而一次性满足以下工程需求：

1. **ECM 模型定制** —— RC 阶数可调，OCV/R0/Rk/Ck/dUdT 支持随 (T, I, SoC) 的查表与解析式，可选 ECMD 扩散过电势
2. **自定义串并联拓扑** —— 标准 `nS nP` 生成器 + 任意手写网表（MNA 修正节点法求解，支持电流/功率控制）
3. **热模型集成** —— 集总热网络 + **三维有限体积热模型**，支持 1D / 2D / 3D 自由切换，
   各向异性导热系数 (kx, ky, kz)、比热容、密度均可自定义
4. **自定义电芯间导热** —— 任意导热矩阵 / 邻接表，隐式欧拉无条件稳定
5. **拓扑热切换（可重构电池包）** —— 运行中随时/定时改变整包拓扑：
   一组工作一段时间后另一组并入并联、故障芯自动旁路等
6. **工业 314Ah 大电芯默认参数** —— 开箱即用的储能电芯规格，含精确几何尺寸与热物性参数

---

## 安装与运行

```bash
cd /workspace
python3.11 ecm_pack/examples/demo.py                  # 四要素演示 → ecm_pack_demo.png
python3.11 ecm_pack/examples/topology_demo.py         # 拓扑热切换：8S 工作 10min 后另一组 8S 并入并联
python3.11 ecm_pack/examples/fault_tolerant_demo.py   # 故障容错：缺陷芯到阈值自动旁路，整包继续放电
python3.11 ecm_pack/examples/demo_314ah_3d.py         # 314Ah 大电芯 1D/2D/3D 热模型对比
python3.11 ecm_pack/examples/demo_8s2p_200a.py        # 8S2P / 总负载 200A 工况 —— 重点考察「环流」
```

库本身无需安装，只要 `ecm_pack/` 在 `PYTHONPATH` 即可 `import ecm_pack as ep`。

依赖：`numpy`, `scipy`, `pandas`, `matplotlib`（均 pip 安装的纯 Python 库）。

---

## 模块结构

| 文件 | 职责 |
|---|---|
| `data.py` | 参数归一化：常量 / 可调用 / 1D·2D·3D 查表统一成可调用函数 |
| `ecm.py` | `ECMCellSpec`（规格）+ `ECMCell`（状态）。RC 支路用**解析指数积分**推进，无条件稳定 |
| `circuit.py` | `Netlist` + `solve_circuit`（MNA 求解，支持电流/功率控制）+ `setup_circuit(nS,nP)` + `setup_two_group` + `setup_series_bypass` |
| `thermal.py` | `ThermalNetwork`：集总热容 + 对流 + **自定义电芯间导热**（隐式欧拉，无条件稳定） |
| `thermal3d.py` | `CellThermalModel`：**三维有限体积热模型**，支持 `dim=1,2,3` 自由切换；各向异性导热 k=(kx,ky,kz)；隐式欧拉 + scipy.sparse 稀疏求解 |
| `defaults.py` | **314Ah 大电芯默认参数**：`cell_314ah_spec()` —— 精确几何 174×71.7×207mm，热物性 ρ=2300, cp=1000, k=(12,0.7,11.6)，电气 R0=R1=0.4mΩ, τ=100s |
| `pack.py` | `Pack`：双步循环耦合 ECM+电路+热；`set_topology()` 与 `solve(topology_events=, switch_callback=)` 支持拓扑热切换 |

---

## 最小用法

### 方式一：手写 ECM 规格（完全自定义）

```python
import numpy as np
import ecm_pack as ep

# 1) 定义一个 1 阶 RC 的 ECM 电芯（参数可随 (T,I,SoC) 变化）
spec = ep.ECMCellSpec(
    capacity=5.0,
    ocv=lambda s: 3.2 + 0.95 * s,                 # OCV(SoC)
    R0=lambda T, I, s: 0.01 * (1 + 0.6 * (1 - s)), # R0(T,I,SoC)
    R=[lambda T, I, s: 0.005], C=[lambda T, I, s: 6000.0],
    dUdT=lambda T, s: -1e-4,                         # 可逆热系数
)

# 2) 2 串 2 并拓扑（也可手写任意网表）
cells = [ep.ECMCell(spec) for _ in range(4)]
netlist, v_rows, ri_rows = ep.setup_circuit(2, 2)

# 3) 热网络（含自定义导热：芯 0-芯 1 强耦合）
thermal = ep.ThermalNetwork(4, C_th=300.0, h=0.5, T_amb=298.15,
                            conduction=[(0, 1, 30.0)])

# 4) 耦合求解：整包 8A 放电 600 步，每步 10s
pack = ep.Pack(cells, netlist, thermal=thermal, v_cut_lower=2.5)
out = pack.solve(dt=10.0, control=8.0, control_type="current", n_steps=600)
print(out["Pack terminal voltage [V]"])
print(out["Cell SoC"])
```

### 方式二：使用 314Ah 大电芯默认参数（开箱即用）

```python
import ecm_pack as ep

# 一行构建 314Ah 储能大电芯（含完整几何与热物性参数）
spec = ep.cell_314ah_spec(soc_init=1.0, T_init=298.15)
cell = ep.ECMCell(spec)

# 查看内嵌的几何 / 热物性（可用于驱动 3D 热模型）
print(spec.Lx, spec.Ly, spec.Lz)   # 0.174  0.0717  0.207  [m]
print(spec.rho, spec.cp, spec.k)   # 2300.0  1000.0  (12.0, 0.7, 11.6)
```

---

## 314Ah 大电芯默认参数

`ep.cell_314ah_spec(soc_init=0.5, T_init=298.15)` 返回一个 **314Ah 储能大电芯** 的完整规格：

| 类别 | 参数 | 值 |
|------|------|----|
| **电气** | 容量 | 314 Ah |
| | OCV 曲线 | LFP 体系（平台特征 ~3.2V） |
| | 欧姆内阻 R0 | **0.4 mΩ** |
| | 极化内阻 R1 | **0.4 mΩ** |
| | RC 时间常数 τ | **100 s**（C1 = τ/R1 = 250 kF） |
| **几何** | 宽度 X | **174 mm** |
| | 厚度 Y | **71.7 mm** |
| | 高度 Z | **207 mm** |
| **热物性** | 密度 ρ | **2300 kg/m³** |
| | 比热容 cp | **1000 J/(kg·K)** |
| | 导热系数 k | **(12, 0.7, 11.6) W/(m·K)** —— 各向异性 |
| | | X(宽度) 12 / Y(厚度) 0.7 / Z(高度) 11.6 |

> 注：厚度方向（Y）导热系数仅 0.7 W/mK，是热管理的瓶颈方向。

---

## 三维热模型：`CellThermalModel`

支持 **1D / 2D / 3D** 的自由切换，通过 `dim` 参数控制：

```python
from ecm_pack.thermal3d import CellThermalModel

# 3D 全维度热模型（默认）
tm3 = CellThermalModel(Lx=0.174, Ly=0.0717, Lz=0.207, dim=3,
                       nx=6, ny=6, nz=10,
                       rho=2300.0, cp=1000.0,
                       k=(12.0, 0.7, 11.6),   # 各向异性
                       h=5.0, T_amb=298.15)

# 1D 模型（仅 X 方向传热，Y/Z 各 1 节点）
tm1 = CellThermalModel(Lx=0.174, Ly=0.0717, Lz=0.207, dim=1)

# 2D 模型（XY 平面，Z 仅 1 节点）
tm2 = CellThermalModel(Lx=0.174, Ly=0.0717, Lz=0.207, dim=2)

# 推进一步：Q 为产热 [W]，dt 步长 [s]
tm3.step(Q=10.0, dt=60.0, t=0.0)

# 获取温度场分布
T_field = tm3.reshape()   # dim=1→(nx,)  dim=2→(nx,ny)  dim=3→(nx,ny,nz)
T_mean, T_max, T_min = tm3.stats()
```

**数值方法**：隐式欧拉（后向欧拉），scipy.sparse 稀疏矩阵求解，任意步长 dt 和导热系数均无条件稳定。

---

## 拓扑热切换（可重构电池包）

核心思想：**电芯的时序状态（SoC/RC/温度）与电路网表完全解耦**。切换拓扑 =
替换网表 + 替换「接入映射 `active`」，不触碰任何电芯的 SoC/RC/温度。因此被断开
的电芯状态冻结、新并入的电芯按自身状态立即参与，并联瞬间由 MNA 自动产生环流/浪涌。

`active` 是「网表第 k 个 V 元素 ↔ cells 中第几个电芯」的映射，**顺序必须与网表
V 元素顺序一致**（见 `examples/topology_demo.py` 中 `active_par` 的踩坑记录）。

三种切换入口：

```python
# ① 定时切换：t=600s 时把整包从 8S 重构成 8S2P（两组 8S 并联）
nl_solo, act_solo, nl_par, act_par = ep.setup_two_group(8)
pack = ep.Pack(cells, nl_solo, active=act_solo)
out = pack.solve(dt=1.0, control=8.0, n_steps=1200,
                 topology_events=[(600.0, nl_par, act_par)])

# ② 事件驱动（随时/按工况）：返回 (netlist, active) 即切换，返回 None 不切换
def cb(t, pack):
    if pack.cells[3].terminal_voltage(0.01, 10.0) < 3.4:   # 缺陷芯逼近截止
        return ep.setup_series_bypass(8, bypass_idx=3)      # 自动旁路该芯
    return None
out = pack.solve(dt=1.0, control=10.0, n_steps=800, switch_callback=cb)

# ③ 运行中随时手动切换
pack.set_topology(nl_par, act_par)   # 立即生效，电芯状态零丢失
```

`setup_two_group(nS)` 返回「单组 nS」与「两组 nS 并联(nS2P)」两套网表；
`setup_series_bypass(n, bypass_idx=k)` 返回 n 串联网表，给定 `k` 时旁路第 k 芯。

---

## 关键设计点

- **RC 解析积分**：`v_rc(t+dt) = (-I·R) + (v_rc - (-I·R))·exp(-dt/τ)`，
  对时间步长无条件稳定，无需刚性 ODE 求解器，整包可固定大步长推进。
- **MNA 耦合**：每个电芯在网表里是「电压源 E_k = OCV+Σv_rc+η_diff」串「电阻 R0_k」。
  解 MNA 得支路电流 I_k，回灌给 ECM 步进 SoC/RC/扩散与产热。
- **隐式热网络**：导热 + 对流用后向欧拉求解，任意大导热系数 G 或步长都不会发散。
- **功率控制**：用 Thevenin 闭式解析解 `I=(V_oc−√(V_oc²−4R_eq·P))/(2R_eq)`，
  无需脆弱迭代，并自动判 `P>Pmax` 不可行。
- **符号约定**：正电流 = 放电（与 PyBaMM 一致）；网表负端接地（node 0），
  电压源正极接电芯正端（b_s），故 `cell_current = -I_batt`。

---

## 验证结果

**测试覆盖**：31 项 pytest 全部通过（`ecm` / `circuit` / `thermal` / `pack` / `thermal3d`）。

**演示图**（`/workspace/*.png`）：

| 场景 | 结果 |
|---|---|
| 单芯 5A 放电 1h | SoC 0.80 → 0，电压 3.95 → 3.07V ✓ |
| 2S2P 不均衡放电 | 并联不均流明显（同串级内电流差 0.08/0.30A）✓ |
| 自定义导热 G=30 | 两芯温度被拉平 298.625/298.625K ✓ |
| 功率控制 120W | 端口功率稳定在 120.0W ✓ |
| 拓扑热切换 8S → 8S2P | 并入瞬间自动产生 ~21.8A 组间环流，10min 内 SoC Δ 0.463→0.078 ✓ |
| 故障容错旁路 | 缺陷芯到 3.4V 自动旁路，整包不宕机（8S→7S，V 28.9→25.5V）✓ |
| 314Ah 1D/2D/3D 热对比 | 1D ΔT=0.19K，2D/3D ΔT~0.74K（厚度方向 k=0.7 W/mK 为瓶颈）✓ |
| **8S2P 200A 环流工况** | **并入瞬间环流峰值 139.7A（I_A=−39.7A 被充电，I_B=239.7A），稳态 10.4A；RC 极化失配主导合闸涌流，后段 SoC 差驱动慢速均衡** ✓ |

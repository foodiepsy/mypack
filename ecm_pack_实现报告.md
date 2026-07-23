# 自研电池包仿真库 `ecm_pack`：用可定制 ECM 复刻 PyBaMM + liionpack 的耦合范式

> 基于上一轮对 PyBaMM（ECM 源码）和 liionpack（电路耦合源码）的研究，
> 这里**不依赖 PyBaMM/liionpack**，用 numpy/scipy 从零搓了一个自包含库，
> 把「可定制 ECM 电芯 + MNA 电路 + 热网络（含电芯间导热）」用双步循环耦合起来。
> 代码位于 `/workspace/ecm_pack/`，演示见 `examples/demo.py`，结果图 `ecm_pack_demo.png`。

---

## 1. 设计范式：从两个库各取所长

| 参考库 | 借鉴的核心思想 | 在本库中的落地 |
|---|---|---|
| **PyBaMM ECM** | 电芯 = `OCV + R0 + Σ(RC) [+ 扩散]`，参数可随 (T,I,SoC) 查表 | `ecm.py`：`ECMCellSpec`/`ECMCell`，RC 用解析指数积分推进 |
| **liionpack** | 每个电芯在网表里是「电压源 + 串联电阻」；用 MNA 解电路得支路电流，回灌电化学模型做一步推进（双步循环） | `circuit.py`：`Netlist`/`solve_circuit`/`setup_circuit`；`pack.py`：`Pack` 驱动循环 |

关键洞察（来自源码精读）：
- liionpack 把电芯当成**黑盒 PyBaMM 模型**，每步用 `I_batt` 当 `Current function [A]` 输入，靠 PyBaMM 内部代数约束求解；本库去掉黑盒，直接用**已知结构的 ECM** 算 OCV/RC/扩散，反而更简单、更快、更可控。
- 电路对电芯来说就是一个**戴维南等效**：电压源 `E_k = OCV+Σv_rc+η_diff`（R0 之后的电动势）串一个电阻 `R0_k`。MNA 解出每支路电流 `I_k`，回灌给 ECM 步进状态。

## 2. 模块与文件

```
/workspace/ecm_pack/
├── data.py        # 参数归一化：常量/可调用/1D·2D·3D 查表 → 统一可调用
├── ecm.py         # ECMCellSpec + ECMCell（RC 解析积分；可选扩散 ECMD）
├── circuit.py     # Netlist + solve_circuit(MNA) + setup_circuit(nS,nP)
├── thermal.py     # ThermalNetwork：集总热容 + 对流 + 自定义电芯间导热
├── pack.py        # Pack：ECM × 电路 × 热网络 的双步耦合循环
├── __init__.py
├── README.md
└── examples/demo.py
```

## 3. 四个需求的落地方式

### 需求 1 · ECM 模型定制（`ecm.py`）
- **RC 阶数可配**：`ECMCellSpec(R=[R1,R2,...], C=[C1,C2,...])`，N 个 RC 支路随意。
- **参数随工况变化**：`R0/Rk/Ck` 接受 `(T[degC], I[A], SoC)` 的函数或 3D 查表；`OCV(SoC)` 1D；`dUdT(T,SoC)` 2D。`data.py` 的 `lookup_1d/lookup_nd` 直接把标定 CSV/数组灌成插值函数。
- **产热**：`Q = I²R0 + Σ(-I·v_rc) + Q_rev`，`Q_rev = -I·T·dUdT`（与 PyBaMM ECM 的 Nieto 拆分一致）。
- **可选扩散 ECMD**：`diffusion=True` 时解 1D 分布 SoC 的 PDE（隐式欧拉 + Thomas 三对角，无条件稳定）。
- **RC 用解析积分**推进：`v(t+dt)=(-IR)+ (v-(-IR))·e^(-dt/τ)` —— 对任意步长**无条件稳定**，免去刚性求解器。

### 需求 2 · 自定义串并联拓扑（`circuit.py`）
- `setup_circuit(nS, nP)` 自动生成标准网格网表（已验证与手写等价）。
- **任意拓扑**：直接传 `Netlist(elements)`，元素 `{'desc','node1','node2','value'}`，`desc` 首字母 `V`=电芯、`R`=电阻、`I`=整包负载。负端必须接地（node 0）。已用一份手写 2s2p 网表验证与生成器结果一致。

### 需求 3 · 热模型集成（`thermal.py` + `pack.py`）
- 每电芯集总热容 `C_th`、对环境对流 `h`（可含随时间变化的环境温度）。
- 每步用 ECM 产热 `Q` 推进：`C_th·dT/dt = Q + Σ_j G_ij(T_j-T_i) - h(T-T_amb)`。
- 热网络用**隐式欧拉**求解 → 任意步长/对流系数都不发散（这是初版显式欧拉踩过的坑）。

### 需求 4 · 自定义电池间相互导热（`thermal.py`）
- `conduction` 支持两种形式：邻接表 `[(i,j,G_ij), ...]` 或完整对称矩阵 `(N,N)`。
- 导热项 `Q_cond = K·T`（K 对角线已置为负和），与产热、对流一并隐式求解。
- **验证**：1s2p 两芯初始 SoC 不同（产热不同），`G=30` 强耦合后两芯温度被精确拉平到 `298.625/298.625 K`，而绝热情形为 `298.633/298.616 K`（保留微小差异）。

## 4. 核心耦合循环（`pack.py::Pack.solve`）

```text
初始化: 把各电芯 E_k(=OCV+Σv_rc)、R0_k 写入网表
循环每一步 dt:
  1) solve_circuit(网表, 控制电流/功率)  →  I_batt (每支路电流)
  2) cell_current_k = -I_batt_k           # 符号调整：正=放电
  3) 每个电芯 step_electrical(I_k, dt)    # 推进 SoC / RC / 扩散，算 R0、产热 Q_k
  4) thermal.step(Q, dt)                  # 产热 + 电芯间导热 + 对流 → 更新 T_k
  5) 把更新后的 E_k、R0_k 写回网表        # 供下一步求解
  6) 记录端口与每芯输出；越电压限则终止
```

这正是 liionpack 的「**更新网表 → 解电路 → 步进电化学模型 → 更新热**」范式，只是把黑盒 PyBaMM 换成了可定制的 ECM。

## 5. 验证结果（`examples/demo.py`，已跑通）

| 场景 | 结果 | 结论 |
|---|---|---|
| 单芯 5A 放电 1h | SoC 0.80→0，电压 3.95→3.07V | 放电极性与 SoC 演化正确 ✓ |
| 2s2p 不均衡放电 | 并联同串级内电流差 0.08/0.30 A | 不均衡荷电→并联不均流 ✓ |
| 自定义导热 G=30 | 两芯温度 298.625/298.625 K（绝热 298.633/298.616） | 电芯间导热拉平温度 ✓ |
| 功率控制 120W | 端口功率稳定 120.0 W | 功率源模式可用 ✓ |

## 6. 与 PyBaMM/liionpack 的差异与取舍

- **优点**：零重型依赖、结构透明、ECM 参数完全可控、热与导热原生支持、单步解析积分快且稳。
- **取舍**：未实现 PyBaMM 的伪二维/全阶电化学（本库聚焦 ECM 层级）；未做 liionpack 的 CasADi/Ray 大规模并行（本库用 numpy 向量化，小到中规模包足够）。
- **可扩展点**：把 `ECMCell` 换成任意 `step(I,dt)` 接口即可接入其他电芯模型；`conduction` 矩阵可接任意热网络拓扑（液冷板、模组夹层等）。

---

**一句话总结**：能搓，而且搓出来的库完整覆盖了「ECM 定制 / 串并联拓扑 / 热集成 / 电芯间导热」四件事，耦合范式与两个参考库一脉相承，已通过物理自洽验证。

# PyBaMM ECM（等效电路模型）实现分析

> 研究对象：`pybamm-team/PyBaMM`（当前仓库已重构为 monorepo，ECM 代码位于 `packages/pybamm/src/pybamm/`）
> 核心入口：`pybamm.equivalent_circuit.Thevenin()`（经典戴维南/Thévenin 等效电路）

---

## 1. 模型架构：用"积木式子模型"拼出电路

PyBaMM 的 ECM 不像传统电池模型那样写一个大方程，而是把电路拆成**若干子模型（submodel）**，每个子模型负责一个电路元件，最后由 `Thevenin` 主模型把它们的变量、方程、事件拼装起来。

`thevenin.py` 中的装配顺序（`set_submodels`，第 194 行）：

| 顺序 | 子模型 | 文件 | 在电路中的角色 |
|---|---|---|---|
| 1 | 外部电路 `External circuit` | `external_circuit/` | 给定电流/电压/功率/电阻，决定边界电流 I |
| 2 | OCV 元件 `Open-circuit voltage` | `ocv_element.py` | 开路电压 + SoC 的演化 |
| 3 | 电阻元件 `Element-0 (Resistor)` | `resistor_element.py` | 串联欧姆内阻 R0 |
| 4 | RC 元件 `Element-N (RC)` × N | `rc_element.py` | N 个并联 RC 支路（极化/弛豫） |
| 5 | 扩散元件 `Diffusion` | `diffusion_element.py` | 可选：分布 SoC 的扩散过电势（ECMD） |
| 6 | 热模型 `Thermal` | `thermal.py` | 双集总：电芯 + 夹具（jig）温度 |
| 7 | 电压模型 `Voltage` | `voltage_model.py` | 把各元件过电势求和得到端电压 V |

```python
def set_submodels(self, build):
    self.set_external_circuit_submodel()   # 1
    self.set_ocv_submodel()                # 2
    self.set_resistor_submodel()           # 3  (固定 1 个 R0)
    self.set_rc_submodels()                # 4  (N 个 RC，N 由选项决定)
    self.set_diffusion_submodel()          # 5
    self.set_thermal_submodel()            # 6
    self.set_voltage_submodel()            # 7
```

### 电路拓扑（默认，1 个 RC 元件）

```
   ┌──────────┐      ┌──────┐      ┌───────────┐
 I →│   OCV    │─┬───│ R0   │─┬───│  RC (R1‖C1) │─┐
   │  U_ocv   │ │   └──────┘ │   └───────────┘ │
   └──────────┘ │            │                 │
                └────────────┴─────────────────┘──→ V
                    （+ 可选扩散过电势 η_diff）
```

端电压由 `VoltageModel.get_coupled_variables`（voltage_model.py 第 23–35 行）直接求和：

```python
overpotential = Σ_i  Element-i overpotential        # 含 R0 与所有 RC 支路
voltage = ocv + overpotential + diffusion_overpotential
```

> **符号约定**：PyBaMM 中**正电流 = 放电**。放电时 `current > 0`，电阻过电势 `-I·R0 < 0`，端电压低于 OCV，符合物理直觉。

---

## 2. 业务逻辑与核心方程（逐元件拆解）

### 2.1 OCV 元件：SoC 是全局状态变量

`ocv_element.py` 中：
- **状态变量**：`SoC`（标量，0~1）
- **开路电压**：`ocv = param.ocv(soc)`，即一个依赖 SoC 的函数/查表
- **SoC 演化（ODE）**，`set_rhs` 第 47–51 行：

  $$\frac{d(\text{SoC})}{dt} = -\frac{I}{Q \cdot 3600}$$

  其中 `Q = cell_capacity [A.h]`，`I` 来自外部电路。

- **可逆热（熵热）**，`get_coupled_variables` 第 33–36 行：

  $$Q_{rev} = -I \cdot T \cdot \frac{dU}{dT}$$

  `dU/dT` 即熵变（entropic change），来自 2D 查表。

- **事件**：SoC 触底/触顶（`Minimum SoC`、`Maximum SoC`）。

### 2.2 电阻元件 Element-0（R0）—— 欧姆内阻

`resistor_element.py`，第 20–38 行：
- 电阻是**依赖温度、电流、SoC 的函数**：`r0 = param.rcr_element("R0 [Ohm]", T, I, soc)`
- 过电势与热：`η0 = -I·R0`，`Q_irr,0 = I²·R0`（焦耳热，恒正）

### 2.3 RC 元件（R1‖C1 等）—— 极化与弛豫动力学

`rc_element.py` 每个 RC 元件：
- **状态变量**：`Element-N overpotential [V]`（即电容上的电压 v_rc）
- 电阻、电容同样来自 3D 查表：`R_N`、`C_N` 依赖 `(T, I, SoC)`，`tau_N = R_N·C_N`
- **RC 动力学 ODE**，`set_rhs` 第 57–66 行：

  $$\frac{d v_{rc}}{dt} = -\frac{v_{rc}}{\tau} - \frac{I \cdot R}{\tau}$$

  稳态时 `v_rc = -I·R`（电容开路，RC 退化为纯电阻）；瞬态按时间常数 τ 弛豫——这就是 ECM 能模拟"负载突变后电压爬升/跌落"的关键。
- 不可逆热：`Q_irr,N = -I·v_rc`

> 默认 `number of rc elements = 1`（1 个 RC 支路）。工业上常用 **2~3 个 RC 支路**分别捕捉快/慢极化（设置 `options={"number of rc elements": 2}` 即可，代码会自动生成 `Element-1`、`Element-2` 等）。

### 2.4 扩散元件（可选，ECMD）—— 高浓度极化

`diffusion_element.py`，由选项 `diffusion element` 开关（默认 `false`）：
- 开启后引入一个**空间分布 SoC** `Distributed SoC`，定义在 `ECMD particle` 一维几何上（`default_geometry`/`default_var_pts` 在 thevenin.py 第 249–268 行定义了 20 点网格）。
- **扩散 PDE**，`set_rhs` 第 85–99 行：

  $$\frac{\partial z}{\partial t} = \frac{1}{\tau_D} \nabla^2 z$$

  边界：`左` Neumann=0，`右` = `-τ_D·I/(Q·3600)`；时间常数 `tau_D = Diffusion time constant [s]`。
- 扩散过电势：`η_diff = -(ocv(z_surf) - ocv(soc))`，即表面 SoC 与体相 SoC 的 OCV 差。

> 这是 ECM 的增强版 **ECMD**（Equivalent Circuit Model with Diffusion，引用 `Fan2022`），用于高倍率下浓度梯度明显的场景。对大多数工况的标准 ECM，可关掉它以省算力。

### 2.5 热模型：电芯 + 夹具双集总

`thermal.py`，最实用的工业特性之一：
- **状态变量**：`Cell temperature [degC]`、`Jig temperature [degC]`
- 环境温度 `T_amb` 可由时间函数给定（`param.T_amb(t)`）
- **热平衡 ODE**，`set_rhs` 第 64–77 行：

  $$\begin{aligned}
  C_{th,cell}\frac{dT_{cell}}{dt} &= Q_{irr} + Q_{rev} - k_{cell,jig}(T_{cell}-T_{jig}) \\
  C_{th,jig}\frac{dT_{jig}}{dt}   &= -k_{jig,air}(T_{jig}-T_{amb}) + k_{cell,jig}(T_{cell}-T_{jig})
  \end{aligned}$$

  其中 `Q_irr = I²·R0 + Σ(-I·v_rc)`，`Q_rev = -I·T·dU/dT`（见 `get_coupled_variables` 第 45–60 行）。
- 这实现了 Nieto 2012 的产热拆分：可逆热 + 不可逆热，并耦合到夹具再向环境散热。

### 2.6 外部电路：工作模式

`thevenin.py` 的 `set_external_circuit_submodel`（第 107–148 行）按选项选择电流来源：
- `current`（默认）：显式电流剖面 `I = current function [A]`
- `voltage` / `power` / `resistance`：解代数方程使端电压/功率/电阻满足目标
- `differential power` / `differential resistance`：用微分方程解功率/电阻
- `CCCV`：恒流恒压充电专用（引用 `Barletta2022thevenin`）
- 可传**自定义 callable** 作为残差约束（FunctionControl）

`ExplicitCurrentControl`（explicit_control_external_circuit.py）直接把 `Current function [A]` 暴露为 `I`，并派生 `C-rate = I / Q`。

### 2.7 终止事件与数值开关

`voltage_model.py` 的 `add_events_from`（第 57–90 行）定义了：
- `Maximum voltage [V]` / `Minimum voltage [V]`：到达上/下限电压（由 `Upper/Lower voltage cut-off [V]` 设定）时**终止求解**
- 两个 `SWITCH` 事件（容差 0.125 V），用于 CasADi "fast with events" 模式平滑切换。

---

## 3. 模型选项（Options）一览

`thevenin.py` 的 `set_options`（第 76–105 行）+ `ecm_model_options.py`：

| 选项 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `number of rc elements` | 自然数 0,1,2,… | 1 | RC 支路数量，决定瞬态精度 |
| `diffusion element` | `true`/`false` | `false` | 是否启用 ECMD 扩散极化（PDE） |
| `operating mode` | `current`/`voltage`/`power`/`differential power`/`resistance`/`differential resistance`/`CCCV`/自定义函数 | `current` | 电流/边界条件施加方式 |
| `calculate discharge energy` | `true`/`false` | `false` | 是否额外计算放电能量/吞吐能量（较贵） |

---

## 4. 构建一个**工业可用 ECM** 需要哪些参数

参数由 `EcmParameters`（`ecm_parameters.py`）定义符号，由 `ParameterValues("ECM_Example")` 提供数值。下面按"必填 / 建议 / 可选"分级。

### 4.1 必填基础参数

| 参数名（PyBaMM key） | 单位 | 含义 | 来源 |
|---|---|---|---|
| `Cell capacity [A.h]` | A·h | 电芯容量（SoC 积分分母） | 容量测试 |
| `Nominal cell capacity [A.h]` | A·h | 标称容量（与上面可能不同） | 规格书 |
| `Initial SoC` | 0~1 | 初始荷电状态 | 工况设定 |
| `Initial temperature [K]` | K | 电芯/夹具初始温度 | 工况设定 |
| `Upper voltage cut-off [V]` / `Lower voltage cut-off [V]` | V | 充/放电截止电压（触发终止事件） | 规格书/BMS |
| `Current function [A]` | A | 电流剖面（或 `Power/Resistance function`） | 工况（应用默认模式） |

### 4.2 电路查表参数（核心，决定精度）

这些都是 **`FunctionParameter`（插值查表）**，是工业 ECM 的"灵魂"。

| 参数名 | 依赖变量 | 维数 | 数据表格式 |
|---|---|---|---|
| `Open-circuit voltage [V]` | SoC | 1D | `SoC, OCV[V]` |
| `R0 [Ohm]` | (温度, 电流, SoC) | **3D** | `T[degC], I[A], SoC, R0` |
| `R1 [Ohm]` … `R_N [Ohm]` | (温度, 电流, SoC) | **3D** | 每个 RC 支路一组 |
| `C1 [F]` … `C_N [F]` | (温度, 电流, SoC) | **3D** | 同 R 一一对应 |
| `Entropic change [V/K]` | (OCV, 温度) | 2D | `OCV[V], T[degC], dUdT` |
| `Element-N initial overpotential [V]` | — | 标量 | RC 支路初始电容电压 |

> ⚠️ **关键工程要点**：`R0/R1/C1` 做成 **3D 查表（T × I × SoC）** 是工业化的分水岭。
> - 仅随 SoC 变化 → 只能跑标定工况；
> - 加上温度 → 覆盖冬夏/热管理；
> - 加上**电流/倍率** → 捕捉大电流下内阻非线性（示例里电流列同时含 ±400 A，即充放电都要覆盖）。
> 查表由 `process_3D_data_csv` 解析：四列、规则网格、C 序（第一列变化最慢=温度，第三列最快=SoC）。

### 4.3 热模型参数（做热仿真/热失控预警必填）

| 参数名 | 单位 | 含义 |
|---|---|---|
| `Cell thermal mass [J/K]` | J/K | 电芯热容 |
| `Cell-jig heat transfer coefficient [W/K]` | W/K | 电芯↔夹具换热系数 |
| `Jig thermal mass [J/K]` | J/K | 夹具热容 |
| `Jig-air heat transfer coefficient [W/K]` | W/K | 夹具↔环境换热系数 |
| `Ambient temperature [K]` | K | 环境温度（可设为时间函数） |

来源：量热法（ARC/HWS）/ 热阻网络拟合。

### 4.4 可选/增强参数

| 参数名 | 单位 | 何时需要 |
|---|---|---|
| `Diffusion time constant [s]` | s | 启用 `diffusion element` 时（ECMD） |
| `RCR lookup limit [A]` | A | 查表电流维的裁剪上限（示例=340 A），防止外推 |
| `calculate discharge energy` 选项 | — | 需要能量/吞吐统计时 |

---

## 5. 示例参数集说明了什么

`input/parameters/ecm/example_set.py` 给了一套**演示用（非真实电芯）**参数，但揭示了工业标定的"配方"：

1. 容量假设 100 A·h；
2. 25°C/50%SoC 下 100 A 直流内阻（DCIR）= 1 mΩ；
3. DCIR 随 **SoC 二次**（两端升到 1.2 mΩ）、**Arrhenius 随温度**（Ea=20000）、**随电流幅值线性**（每 100 A 斜率 0.01 mΩ）变化；
4. `R0 = 40%·DCIR`，`R1 = 60%·DCIR`，`C1 = τ/R1`（τ=30 s）；
5. OCV、dU/dT 取自文献（未公开来源）。

> 这套逻辑可以直接照搬为你的**实验标定流程**：用 HPPC 脉冲测 DCIR → 拆分为 R0/R1 → 由弛豫曲线拟合 τ 和 C → 扫温度/SoC/倍率生成 3D 表。

数据表在 `input/parameters/ecm/data/`：`ecm_example_ocv.csv`、`ecm_example_r0.csv`、`ecm_example_r1.csv`、`ecm_example_c1.csv`、`ecm_example_dudt.csv`。

---

## 6. 一个工业级 ECM 的"最小可行"构建清单

1. **确定 RC 阶数**：先跑 1 阶，用脉冲+弛豫数据看残差，必要时升到 2~3 阶。
2. **标定 OCV-SoC 曲线**：低倍率（C/20~C/50）充放电信步法，1D 查表。
3. **HPPC 标定 R0/R1/C1**：在若干 SoC 点、若干温度下做脉冲，得到 R/C；再扩展到多个倍率点 → 拼成 3D 查表。
4. **熵热系数 dU/dT**：等温熵热法/弛豫电压法，2D 查表。
5. **热参数**：量热实验得到 `C_th`、各换热系数 `k`。
6. **容量与截止电压**：取自规格书/BMS 限值。
7. **选模式**：实车/BMS 通常用 `current`（电流剖面由外环给定）；台架 CV/CCCV 用 `voltage`/`CCCV`。
8. **（可选）ECMD**：高倍率持续大电流（>3C）且出现明显"电压回弹"时启用扩散元件。

### 典型调用

```python
import pybamm

# 单 RC + 默认电流模式（工业常用配置）
model = pybamm.equivalent_circuit.Thevenin(
    options={
        "number of rc elements": 2,        # 2 个 RC 支路提精度
        "diffusion element": "false",
        "operating mode": "current",
    }
)

# 关键：用你自己的 3D 查表替换示例参数
params = pybamm.ParameterValues("ECM_Example")   # 替换为实测标定集
params["Cell capacity [A.h]"] = 120
params["Current function [A]"] = ...              # 你的工况剖面

# 或用设定 SOC 初始化
params = pybamm.equivalent_circuit.set_initial_state("3.7 V", params)

sim = pybamm.Simulation(model, parameter_values=params)
sim.solve()
```

示例脚本见 `examples/scripts/run_ecm.py`（含完整 `pybamm.Experiment` 充放电循环）与 `examples/scripts/run_ecmd.py`（带扩散元件的 ECMD 版本）。

---

## 7. 实现亮点（工程角度）

- **元件可插拔**：想加 RC 支路、开扩散、换工作模式，只改 options，不动核心方程——子模型架构天然解耦。
- **查表即模型**：R/C/OCV/dUdT 全部是 `FunctionParameter`，意味着**模型结构固定、精度由数据决定**，非常适合把实验室标定数据直接灌进去。
- **产热=可逆+不可逆**的双集总热模型，直接对接热管理与安全预警，这是很多开源 ECM 缺失的。
- **事件终止 + CasADi 快速模式**，适合大规模参数扫描/实验设计。

---

### 主要源码索引

| 文件 | 职责 |
|---|---|
| `models/full_battery_models/equivalent_circuit/thevenin.py` | 主模型、选项、装配 |
| `models/submodels/equivalent_circuit_elements/ocv_element.py` | OCV + SoC ODE + 熵热 |
| `models/submodels/equivalent_circuit_elements/resistor_element.py` | R0 欧姆内阻 |
| `models/submodels/equivalent_circuit_elements/rc_element.py` | RC 支路动力学 |
| `models/submodels/equivalent_circuit_elements/diffusion_element.py` | ECMD 扩散（PDE） |
| `models/submodels/equivalent_circuit_elements/thermal.py` | 双集总热模型 |
| `models/submodels/equivalent_circuit_elements/voltage_model.py` | 端电压求和 + 截止事件 |
| `parameters/ecm_parameters.py` | 参数符号定义 |
| `input/parameters/ecm/example_set.py` + `data/*.csv` | 示例参数与查表数据 |
| `models/full_battery_models/equivalent_circuit/initial_state.py` | 按电压/SoC 初始化 |

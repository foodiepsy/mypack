# mypack

一个**自包含、零重型依赖**（仅 numpy / scipy / pandas / matplotlib）的电池包仿真库，
基于 PyBaMM 的等效电路模型（ECM）思想与 liionpack 的「电路-电芯耦合」范式实现。

核心包为 [`ecm_pack/`](./ecm_pack)，同时满足以下工程需求：

1. **ECM 模型定制** —— Thévenin 等效电路，RC 阶数可调，OCV/R0/Rk/Ck/dUdT 支持随 (T, I, SoC) 的查表与解析式，可选 ECMD 扩散过电势
2. **自定义串并联拓扑** —— 标准 `nS nP` 生成器 + 任意手写网表（MNA 修正节点法求解，支持电流/功率控制）
3. **热模型集成** —— 集总热网络 + **三维有限体积热模型（1D/2D/3D 切换）**，各向异性导热 k=(kx,ky,kz)；**非对称冷却**（每面独立 h）；**壳层热阻**（T_surface 对标 BMS 传感器）
4. **自定义电芯间导热 + 层间热接触电阻** —— 任意导热矩阵 / 邻接表 / R_th 界面热阻（硅胶垫/气隙效应），隐式欧拉无条件稳定
5. **拓扑热切换（可重构电池包）** —— 运行中随时/定时改变整包拓扑：一组工作→另一组并入并联、故障芯自动旁路
6. **工业 314Ah 大电芯默认参数** —— `cell_314ah_spec()` 开箱即用（R0=R1=0.4mΩ, τ=100s, 174×71.7×207mm, ρ=2300, cp=1000, k=(12,0.7,11.6)）
7. **接触/连接电阻建模** —— `R_contact` 参数模拟 tab/焊接/busbar 等效串联电阻的附加压降与发热

## 快速开始

```bash
git clone https://github.com/foodiepsy/mypack.git
cd mypack
# 6 个 demo，每个独立文件夹，产物在各自的 result/ 子目录
python3.11 ecm_pack/examples/demo1-8s-foam/demo_8s_foam_thermal.py    # 8S 大面背靠背 + 薄侧泡棉 + 三维热模型
python3.11 ecm_pack/examples/demo2-8s2p/demo_8s2p_200a.py             # 8S2P / 200A 环流工况分析
python3.11 ecm_pack/examples/demo3-314ah-3d/demo_314ah_3d.py          # 314Ah 大电芯 1D/2D/3D 热模型对比
python3.11 ecm_pack/examples/demo4-basic/demo.py                       # ECM + 拓扑 + 热模型 + 导热（四要素）
python3.11 ecm_pack/examples/demo5-fault-tolerant/fault_tolerant_demo.py  # 故障容错：缺陷芯到阈值自动旁路
python3.11 ecm_pack/examples/demo6-topology/topology_demo.py           # 拓扑热切换：运行中并入另一组 8S
```

只要 `ecm_pack/` 在 `PYTHONPATH` 即可 `import ecm_pack as ep`。

## 测试

```bash
python3.11 -m pytest tests/ -v    # 51 项测试全部通过
```

## 目录

| 路径 | 说明 |
|---|---|
| `ecm_pack/` | 仿真库本体（见其 [`README.md`](./ecm_pack/README.md) 获取完整 API 与参数说明） |
| `ecm_pack/examples/` | 六个演示脚本，按 `demo1-8s-foam/` ~ `demo6-topology/` 分文件夹，产物在各自的 `result/` |
| `ecm_pack/examples/demo_old/` | 旧版 demo1（8s 泡棉）和 demo2（8s2p）的历史脚本与产物（归档，不再维护） |
| `PyBaMM_ECM_实现分析.md` | PyBaMM ECM 源码分析（工业级 ECM 参数清单） |
| `ecm_pack_实现报告.md` | 本库实现报告 |
| `拓扑热切换_实现说明.md` | 拓扑热切换（可重构电池包）原理、验证与踩坑记录 |
| `REVIEW.md` | 项目代码审查记录（含已修复的缺陷清单） |

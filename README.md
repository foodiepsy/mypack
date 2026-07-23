# mypack

一个**自包含、零重型依赖**（仅 numpy / scipy / pandas / matplotlib）的电池包仿真库，
基于 PyBaMM 的等效电路模型（ECM）思想与 liionpack 的「电路-电芯耦合」范式实现。

核心包为 [`ecm_pack/`](./ecm_pack)，同时满足以下工程需求：

1. **ECM 模型定制** —— Thévenin 等效电路，RC 阶数可调，OCV/R0/Rk/Ck/dUdT 支持随 (T, I, SoC) 的查表与解析式，可选 ECMD 扩散过电势
2. **自定义串并联拓扑** —— 标准 `nS nP` 生成器 + 任意手写网表（MNA 修正节点法求解，支持电流/功率控制）
3. **热模型集成** —— 集总热网络 + **三维有限体积热模型（1D/2D/3D 自由切换）**，各向异性导热系数 k=(kx, ky, kz)
4. **自定义电芯间导热** —— 任意导热矩阵 / 邻接表，隐式欧拉无条件稳定
5. **拓扑热切换（可重构电池包）** —— 运行中随时/定时改变整包拓扑：一组工作一段时间后另一组并入并联、故障芯自动旁路
6. **工业 314Ah 大电芯默认参数** —— `cell_314ah_spec()` 开箱即用（R0=R1=0.4mΩ, τ=100s, 174×71.7×207mm, ρ=2300, cp=1000, k=(12, 0.7, 11.6)）

## 快速开始

```bash
git clone https://github.com/foodiepsy/mypack.git
cd mypack
python3.11 ecm_pack/examples/demo.py                  # 四要素演示
python3.11 ecm_pack/examples/topology_demo.py         # 拓扑热切换：8S 工作 10min 后另一组 8S 并入并联
python3.11 ecm_pack/examples/fault_tolerant_demo.py   # 故障容错：缺陷芯到阈值自动旁路
python3.11 ecm_pack/examples/demo_314ah_3d.py         # 314Ah 大电芯 1D/2D/3D 热模型对比
python3.11 ecm_pack/examples/demo_8s2p_200a.py        # 8S2P / 总负载 200A 环流工况分析
```

只要 `ecm_pack/` 在 `PYTHONPATH` 即可 `import ecm_pack as ep`。

## 测试

```bash
python3.11 -m pytest tests/ -v    # 31 项测试全部通过
```

## 目录

| 路径 | 说明 |
|---|---|
| `ecm_pack/` | 仿真库本体（见其 [`README.md`](./ecm_pack/README.md) 获取完整 API 与参数说明） |
| `ecm_pack/examples/` | 五个演示脚本与生成的对比图 |
| `recognize pack 8k/` | 8S2P / 200A 环流工况的完整结果（图、CSV 数据、脚本、README） |
| `PyBaMM_ECM_实现分析.md` | PyBaMM ECM 源码分析（工业级 ECM 参数清单） |
| `ecm_pack_实现报告.md` | 本库实现报告 |
| `拓扑热切换_实现说明.md` | 拓扑热切换（可重构电池包）原理、验证与踩坑记录 |
| `REVIEW.md` | 项目代码审查记录（含已修复的缺陷清单） |

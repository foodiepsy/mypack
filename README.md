# mypack

一个**自包含、零重型依赖**（仅 numpy / scipy / pandas / matplotlib）的电池包仿真库，
基于 PyBaMM 的等效电路模型（ECM）思想与 liionpack 的「电路-电芯耦合」范式实现。

核心包为 [`ecm_pack/`](./ecm_pack)，同时满足五类工程需求：

1. **ECM 模型定制** —— Thévenin 等效电路，RC 阶数可调，OCV/R0/Rk/Ck/dUdT 支持随 (T,I,SoC) 的查表与解析式，可选 ECMD 扩散过电势；
2. **自定义串并联拓扑** —— 标准 `nS nP` 生成器 + 任意手写网表（MNA 修正节点法求解）；
3. **热模型集成** —— 每芯集总热容 + 对环境对流 + 产热（欧姆热 + 可逆熵热）；
4. **自定义电芯间导热** —— 任意导热矩阵 / 邻接表，隐式欧拉无条件稳定；
5. **拓扑热切换（可重构电池包）** —— 运行中随时/定时改变整包拓扑：一组工作一段时间后另一组并入并联、故障芯自动旁路等。

## 快速开始

```bash
git clone https://github.com/foodiepsy/mypack.git
cd mypack
python3.11 ecm_pack/examples/demo.py                # 四要素演示
python3.11 ecm_pack/examples/topology_demo.py       # 拓扑热切换：8S 工作 10min 后另一组 8S 并入并联
python3.11 ecm_pack/examples/fault_tolerant_demo.py # 故障容错：缺陷芯到阈值自动旁路，整包继续放电
```

只要 `ecm_pack/` 在 `PYTHONPATH` 即可 `import ecm_pack as ep`。

## 目录

| 路径 | 说明 |
|---|---|
| `ecm_pack/` | 仿真库本体（见其 [`README.md`](./ecm_pack/README.md) 获取完整 API 与示例） |
| `ecm_pack/examples/` | 三个演示脚本与生成的对比图 |
| `PyBaMM_ECM_实现分析.md` | PyBaMM ECM 源码分析（需要的工业级 ECM 参数清单） |
| `ecm_pack_实现报告.md` | 本库实现报告 |
| `拓扑热切换_实现说明.md` | 拓扑热切换（可重构电池包）原理、验证与踩坑记录 |

# 批量与性能

**中文** | [English](../en/batch-and-performance.md)

## 批量语义

`featurize_batch` 是 `smiles2unimolitem` 的并行外壳，保证以下语义：

- **等长保序**：输出列表与输入一一对应、顺序不变，每个输入恰好一条结果。
- **重复不折叠**：重复的 SMILES 各占一行（不去重）。
- **单行隔离**：单行抛出的未预期异常被捕获为该行的 `status="error"`，整批继续。
- **有界提交**：滑动窗口限制在飞的 future 数量，不会一次把所有任务提交完；输入与结果列表仍各占 `N` 项内存。
- **`workers=1` 纯串行**：不走线程池，没有额外调度开销。
- **参数预校验**：非法 `coord_mode` / `on_timeout` / `timeout` / `workers` / `window` 在提交任何任务之前抛 `ValueError`（空输入也会校验）。

## 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `workers` | `min(8, cpu)` | 线程数；`1` 走串行路径；`<= 0` 抛 `ValueError` |
| `timeout` | `30.0` | 每个构象的墙钟预算（embed + MMFF）；`None` 关闭 |
| `on_timeout` | `"fallback_2d"` | `"fallback_2d"` / `"error"` / `"skip"` |
| `coord_mode` | `"auto"` | 逐行转发给 `smiles2unimolitem` |
| `rng_seed` | `42` | 构象采样与原子截断的种子 |
| `max_atoms` | `256` | 原子数上限 |
| `tokenizer` | `None` | 共享 tokenizer（只读）；`None` 用默认实例 |
| `window` | `max(workers * 4, 16)` | 在飞任务窗口大小 |

## 超时与错误

- 批量层内部始终以 `return_meta=True` 调用单分子 API，因此 `on_timeout="error"` 的超时表现为该行 `status="timeout_error"`、`item=None`，**不会中断整批**，也不需要捕获 `FeaturizationTimeout`。
- `on_timeout="skip"` 的超时行为 `status="skipped"`；`fallback_2d` 行为 `status="degraded_2d"` 且仍返回 `item`。
- 非法 SMILES 行为 `status="invalid"`；未预期异常行为 `status="error"`，`reason` 为异常类型名。
- 单行异常隔离意味着**结果里可能混有 `item=None` 的行**，下游必须按 `meta["status"]` 过滤。

## 性能基准

125 个 HMDB 分层样本（重原子 5 桶 × 25，32 vCPU）：

| workers | 墙钟 | 加速比 |
| --- | --- | --- |
| 1 | 27.82 s | 1.00× |
| 4 | 8.73 s | 3.19× |
| 8 | 5.56 s | **5.00×** |

- **8 线程是甜点**：16 线程效率降到约 52%，因此默认 `workers=min(8, cpu)`。
- 该样本中 **33/125 条长链脂类 `embed_failed → 2D`**，与改造前逐位比对一致，不是本库引入的回归。
- 加速比止步于 5× 的主因是 2D 路径的张量化：那一步是 Python/NumPy 层操作，受 GIL 限制，线程并行无法完全摊开。

RDKit 的 embed / MMFF 会释放 GIL，所以这里用**线程池**而不是进程池；进程池只会额外付出序列化与内存复制的成本。

## 复跑基准

```shell
python benchmarks/bench_featurize.py --csv <csv> --per-bin 25 --workers 1,4,8
```

`--csv` 需要含 `SMILES` 列，按重原子分桶（`[5,15)`、`[15,25)`、`[25,40)`、`[40,60)`、`[60,80)`）各取前 `--per-bin` 行。脚本只打印耗时，不做阈值断言。不带 `--csv` 时跑内置的小样本冒烟集。

## 确定性边界

- 随机性全部来自局部 `np.random.RandomState`，同一输入 + 同一种子在单线程与多线程下逐位一致。
- 结果绑定 RDKit 版本（当前验证 `2025.09.1`）。
- **启用 `timeout` 时跨机器可能不一致**：同一分子在有预算的机器上可能 3D 成功，在慢机器上降级 2D。需要严格可复现时用 `timeout=None`，或按 `meta["status"]` 过滤。

## 何时用批量层

- 数据量大、单分子耗时波动大（含长链脂类等易嵌入失败的分子）时，`featurize_batch(..., timeout=30.0)` 能保证坏分子不拖垮整批。
- 需要逐行审计（哪些分子降级 2D、哪些解析失败）时，`meta` 已经给出答案，不必自己包 try/except。
- 只有很少几个分子、或调用方已有自己的调度器时，直接调 `smiles2unimolitem` 即可。

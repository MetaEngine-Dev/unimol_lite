# 批量与运行时

**中文** | [English](../en/04-batch-and-runtime.md) · [返回主页](../../README.md#zh-cn)

**层级：** 2 — 使用

**前置条件：** [特征化 API](./03-featurization.md)。

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

- 批量层始终以 `return_meta=True` 调用单分子 API，因此 `on_timeout="error"` 的超时表现为该行 `status="timeout_error"`、`item=None`，**不会中断整批**，也不需要捕获 `FeaturizationTimeout`。
- `on_timeout="skip"` 的超时行为 `status="skipped"`；`fallback_2d` 行为 `status="degraded_2d"` 且仍返回 `item`。
- 非法 SMILES 行为 `status="invalid"`；未预期异常行为 `status="error"`，`reason` 为异常类型名。
- 单行异常隔离意味着**结果里可能混有 `item=None` 的行**，下游必须按 `meta["status"]` 过滤。

## 运行时

- 默认 `workers=min(8, cpu)`；线程数取决于硬件与分子类型；
- 大分子可能因 3D 构象生成失败而自动降级 2D（`status="degraded_2d"`）；
- 耗时取决于硬件、分子类型与线程数；可在本机运行 `benchmarks/bench_featurize.py` 测量。

## 确定性边界

- 同一输入与种子在串行/多线程下结果一致；
- 结果依赖 RDKit 版本（已验证 `2025.09.1`）；
- 启用 `timeout` 时，慢机器上可能降级 2D；需要严格可复现时用 `timeout=None`，或按 `meta["status"]` 过滤。

## 何时使用批量层

- 数据量大、单分子耗时波动大（含长链脂类等易嵌入失败的分子）时，`featurize_batch(..., timeout=30.0)` 可避免单个坏分子阻塞整批。
- 需要逐行审计（哪些分子降级 2D、哪些解析失败）时，`meta` 已给出状态，无需额外捕获异常。
- 分子数量很少，或调用方已有调度器时，可直接调用 `smiles2unimolitem`。

**下一页：** [模型、迁移与 FAQ](./05-model-and-faq.md)

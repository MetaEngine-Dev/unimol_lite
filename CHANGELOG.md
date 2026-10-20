# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的组织方式，版本号在正式发布前暂为 `0.0.0.dev`。

## [Unreleased]

### Added

- `smiles2coords` / `smiles2unimolitem` 新增 `coord_mode`（`"auto"` / `"3d"` / `"2d"`）、`timeout`、`on_timeout`（`"fallback_2d"` / `"error"` / `"skip"`）与 `return_meta` 参数。
- 新增 `meta` 状态模型：`status`（`ok` / `degraded_2d` / `timeout_error` / `invalid` / `skipped` / `error`）、`coord_mode`、`reason`、`elapsed_s`。
- 新增批量入口 `featurize_batch(smiles, workers=..., timeout=30.0, ...)`：线程池并行、与输入**等长保序**、有界提交、默认 `workers=min(8, cpu)`，每项返回 `(item | None, meta)`；`unimol_lite` 与 `unimol_lite.data` 均导出。
- 新增 `FeaturizationTimeout` 异常（`on_timeout="error"` 且 `return_meta=False` 时抛出）。
- 新增线程安全的局部 RNG 工具 `make_rng` / `derive_seed`（`unimol_lite/data/utils.py`）。
- 新增吞吐基准脚本 `benchmarks/bench_featurize.py`（自包含冒烟样本 + `--csv --per-bin` 分层基准）。
- 新增 `CHANGELOG.md` 与 `LICENSE`（MIT）。

### Changed

- `use_3d` 弃用：映射到 `coord_mode`（`True` → `"3d"`，`False` → `"auto"`）并发 `DeprecationWarning`；同时显式传 `coord_mode` 时以它为准。
- **默认坐标模式变化**：`coord_mode` 默认 `"auto"`，即 ≤400 原子走 3D、>400 原子走 2D（旧默认 `use_3d=True` 对所有尺寸走 3D）。依赖超大分子 3D 结果的调用方需显式 `coord_mode="3d"`。
- 特征化不再读写全局 `np.random` 状态，改用每次调用创建的局部 `np.random.RandomState`（与旧全局 seed 序列逐位一致）。
- `EmbedMolecule` 失败不再进入历史上 `res == -1` 的 5000 次无超时重试：失败/超时直接按 `on_timeout` 处理，且不做二次重试。
- `timeout` 路径改用 `rdDistGeom.EmbedParameters`（`timeout` 为整数秒、`maxIterations=0`、显式恢复旧 ETKDG 开关），MMFF 优化纳入同一墙钟预算；超预算或失败按 `on_timeout` 降级。
- 2D 降级在 `deepcopy` 的干净分子上计算，且每个构象返回独立数组，避免残留构象或数组别名污染。
- 批量层对非法参数（`coord_mode` / `on_timeout` / `timeout` / `workers` / `window`）在提交前抛 `ValueError`；单行未预期异常隔离为 `status="error"`，不中断整批。
- `README` 重写：安装地址改为私有 SSH 仓库（PyPI 发布 TBD）、明确 featurization 不需要权重、新增「分子采样」章节与 `use_3d` 迁移对照。

### Performance

- 125 个 HMDB 分层样本（重原子 5 桶 × 25）：串行 27.82 s → 4 线程 8.73 s（3.19×）→ **8 线程 5.56 s（5.00×）**；8 线程为甜点（16 线程效率约 52%）。
- 该样本中 33/125 条长链脂类 `embed_failed → 2D`，与改造前逐位比对一致（非回归）；2D 路径张量化受 GIL 限制，是加速比止步 5× 的主因。
- 启用 `timeout` 后单个坏分子不再拖慢整批（超时在预算内返回并降级/报错/跳过）。

### Tests

- 新增 golden 快照对比（小/中/大分子、>400 原子 2D 路径、>256 原子截断路径），改造后逐位一致。
- 新增 `tests/test_coord_mode.py`（`coord_mode` 语义、`2d` 不调用 `EmbedMolecule`、参数校验、`use_3d` 兼容与告警）。
- 新增 `tests/test_rng_compat.py`（局部 RNG 与旧全局序列逐位一致、进程无关 seed 派生、全局状态不被修改）。
- 新增 `tests/test_timeout.py`（真实超时、无二次重试、MMFF 预算、三种 `on_timeout`、`timeout=None` 与有限预算逐位一致）。
- 新增 `tests/test_batch.py`（保序等长、重复 SMILES、串行与 8 线程逐位一致、超时/异常逐行隔离、非法批量参数 `ValueError`）。
- 全量测试：**89 passed, 1 skipped**（skip 为无本地预训练权重的 `test_model_from_pretrained`）。
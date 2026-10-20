# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的组织方式，当前版本为 `0.1.0`（tag `v0.1.0`）。

## [0.1.0]

### Added

- `smiles2coords` / `smiles2unimolitem` 新增 `coord_mode`（`"auto"` / `"3d"` / `"2d"`）、`timeout`、`on_timeout`（`"fallback_2d"` / `"error"` / `"skip"`）与 `return_meta`。
- 新增 `meta` 状态：`status`（`ok` / `degraded_2d` / `timeout_error` / `invalid` / `skipped` / `error`）、`coord_mode`、`reason`、`elapsed_s`。
- 新增批量入口 `featurize_batch(...)`：并行特征化、与输入等长保序、默认 `workers=min(8, cpu)`，每项返回 `(item | None, meta)`。
- 新增 `FeaturizationTimeout` 异常（`on_timeout="error"` 且 `return_meta=False` 时抛出）。
- 新增中英双语文档 `docs/`，根 README 增加文档导航。
- 新增 `CHANGELOG.md` 与 `LICENSE`（MIT）。

### Changed

- `use_3d` 弃用：映射到 `coord_mode`（`True` → `"3d"`，`False` → `"auto"`）并发出 `DeprecationWarning`。
- 默认坐标模式为 `"auto"`：≤400 原子走 3D，>400 原子走 2D。依赖超大分子 3D 结果的调用方需显式 `coord_mode="3d"`。
- 特征化不再读写全局 `np.random` 状态，每次调用使用独立的随机数生成器。
- 3D 构象失败或超时按 `on_timeout` 处理，不再进行无界重试。
- `timeout` 覆盖构象生成与 MMFF 优化；超预算同样按 `on_timeout` 降级/报错/跳过。
- 批量层对非法参数（`coord_mode` / `on_timeout` / `timeout` / `workers` / `window`）在提交前抛 `ValueError`；单行异常隔离为 `status="error"`，不中断整批。

### Performance

- 在 32 vCPU 机器上，125 个代表性分子：串行 27.82 s → 4 线程 8.73 s（3.19×）→ 8 线程 5.56 s（**5.00×**）。
- 大分子可能因 3D 构象生成失败而自动降级 2D。
- 启用 `timeout` 后，单个坏分子不会拖慢整批。

### Tests

- 新增 `coord_mode`、超时/降级、局部 RNG、批量并行与回归测试。

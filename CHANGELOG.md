# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-10-22

### Changed

- 文档重构为编号式 `docs/{en,zh-CN}/`，新增文档总览与语言索引页；每页带 Level/Prerequisites/Next。
- README 改为英文优先的双语主页；恢复并突出 Uni-Mol 论文引用，并写入 `CITATION.cff` 与 `NOTICE`。
- 新增 `AGENTS.md`、`AUTHORS`、`NOTICE`、`CITATION.cff` 与 `Makefile`。
- 项目元数据只保留 `authors`，移除 `maintainers`；`license` 改为 LICENSE 文件，补充 classifiers、项目主页与 sdist 文件清单。
- 统一 Python 版本说明为 `>=3.10,<3.12`，并注明与 `smiles2class` 的组合安装范围一致。

## [0.2.1] - 2026-10-22

### Changed

- 安装说明指向 `v0.2.1`。

## [0.2.0] - 2026-10-22

### Added

- 新增 `unimol_lite.training`：掩码原子预训练的数据变换、LMDB 数据集、PyTorch Lightning 模块与损失。
- 新增 CLI：`unimol-lite prepare-lmdb`、`unimol-lite featurize`、`unimol-lite pretrain`、`unimol-lite selftest`。
- 新增 `train` 可选依赖（`pytorch-lightning`、`pyyaml`、`safetensors`）。
- 新增中英双语[预训练](docs/zh-CN/06-training.md)、[命令行接口](docs/zh-CN/07-cli.md)与[引用与声明](docs/zh-CN/09-references.md)文档。

### Changed

- `UniMolConfig` 新增 `x_norm_loss` 字段。
- 模型在无 padding 的单分子输入下也能计算坐标预训练分支。
- 预训练导出为 Hugging Face 风格的 `pretrained/` 目录，`config.json` 的 `mode` 为 `infer`。
- 版本号更新为 `0.2.0`。

## [0.1.1] - 2026-10-21

### Changed

- 统一 README 与文档中的安装、运行时说明。

## [0.1.0] - 2026-10-21

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

### Runtime

- 大分子可能因 3D 构象生成失败而自动降级 2D。
- 启用 `timeout` 后，单个分子的构象生成受预算约束。

### Tests

- 新增 `coord_mode`、超时/降级、局部 RNG、批量并行与回归测试。

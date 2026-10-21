# UniMol-Lite 概览

**中文** | [English](../en/index.md)

[Uni-Mol](https://github.com/deepmodeling/Uni-Mol) 的轻量实现：一条从 SMILES 到模型输入的完整通路（RDKit 构象生成 + 张量化），提供前向模型 API 与掩码原子预训练。

## 这是什么

UniMol-Lite 把「SMILES → 模型输入」拆成两层：

- **特征化**（`smiles2unimolitem` / `featurize_batch`）：RDKit 生成 3D 构象并做 MMFF 优化，再转成 `src_tokens` / `src_coord` / `src_distance` / `src_edge_type` 四个张量。
- **模型**（`UniMolModel`）：与 Uni-Mol 一致的 encoder / pair 表示。

## 核心特性

- **特征化开箱即用**：只依赖 RDKit。
- **坐标模式可控**：`coord_mode="auto" | "3d" | "2d"`；`auto` 对 ≤400 原子走 3D、更大分子走 2D。
- **超时可预期**：每个构象一份墙钟预算，`on_timeout` 决定降级 2D、报错还是跳过。
- **状态可追踪**：`return_meta=True` 拿到 `status` / `coord_mode` / `reason` / `elapsed_s`。
- **批量并行**：`featurize_batch` 用线程池，等长保序、重复 SMILES 不折叠、单行异常隔离。
- **确定性**：随机性来自局部 `np.random.RandomState`，结果绑定 RDKit 版本。

## 文档地图

| 页面 | 你会在里面找到 |
| --- | --- |
| [快速开始](getting-started.md) | 安装、最小可运行示例 |
| [特征化 API](featurization.md) | 全部函数签名、`coord_mode`、`timeout`、`meta` |
| [批量与运行时](batch-and-runtime.md) | 批量语义、参数、确定性边界 |
| [模型、迁移与 FAQ](model-and-faq.md) | 模型 API、`use_3d` 迁移、常见问题、开发测试 |
| [预训练](training.md) | 掩码原子预训练、数据准备、配置、输出 |
| [命令行接口](cli.md) | `unimol-lite` 子命令与参数 |
| [引用与声明](references.md) | 官方仓库、论文引用、许可证与非官方声明 |

## 版本与许可

当前版本 `0.2.0`，许可证 MIT。代码源自 [Uni-Mol](https://github.com/deepmodeling/Uni-Mol)（DP Technology，MIT）。本仓库不是 Uni-Mol 官方版本；需要官方实现或权重请前往官方仓库。

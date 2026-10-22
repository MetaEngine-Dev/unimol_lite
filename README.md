# UniMol-Lite

[中文](#zh) | [English](#en)

<a id="zh"></a>

## 中文

UniMol-Lite 是 [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) 的轻量实现：将 SMILES 转换为 Uni-Mol 模型输入（RDKit 构象生成 + 张量化），并提供前向模型 API、掩码原子预训练与命令行工具。

> 本项目是独立第三方实现，**不是** Uni-Mol 官方版本。官方实现、权重与支持见 [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol)。

### 安装

Python `>=3.10,<3.12`。

```shell
pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

已配置 SSH 密钥时，也可使用：

```shell
pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

开发与预训练依赖：

```shell
pip install -e ".[dev]"
pip install -e ".[train]"
```

### 快速开始

单个分子：

```python
from unimol_lite.data import smiles2unimolitem

item, meta = smiles2unimolitem(
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    coord_mode="auto",
    timeout=30.0,
    return_meta=True,
)
print(meta["status"], meta["coord_mode"])
print({key: tuple(value.shape) for key, value in item.items()})
```

批量：

```python
from unimol_lite import featurize_batch

results = featurize_batch(["CCO", "not-a-smiles"], workers=4, timeout=30.0)
for item, meta in results:
    shape = None if item is None else tuple(item["src_tokens"].shape)
    print(meta["status"], shape)
```

`item` 包含 `src_tokens`、`src_coord`、`src_distance` 与 `src_edge_type` 四个张量。批量结果与输入等长保序，每项附带 `meta`：

| `status` | 含义 |
|---|---|
| `ok` | 正常 |
| `degraded_2d` | 3D 失败或超时，已降级 2D |
| `invalid` | SMILES 无法解析 |
| `timeout_error` | 超时且 `on_timeout="error"` |
| `skipped` | 超时且 `on_timeout="skip"` |
| `error` | 批量层捕获的未预期异常 |

### 模型

```python
from unimol_lite.model import UniMolModel

model = UniMolModel.from_pretrained("runs/pretrain/pretrained")
model.eval()
encoder_rep, pair_rep = model(**batch)
# encoder_rep: [B, L, D]
# pair_rep:    [B, L, L, D]
```

### 预训练与命令行

```shell
# 数据准备
unimol-lite prepare-lmdb --input input.csv --output molecules.lmdb --smiles-column smiles --nproc 8

# 单分子特征化
unimol-lite featurize --smiles "CCO"

# 掩码原子预训练
unimol-lite pretrain --lmdb molecules.lmdb --output-dir runs/pretrain --config pretrain.yaml

# 前向/损失自检
unimol-lite selftest
```

### 文档导航

主页只保留安装与最小示例；API、运行时、预训练等细节在对应页面展开。建议从“入门”开始：

| 阶段 | 页面 | 内容 |
|---|---|---|
| 入门 | [快速开始](docs/zh/getting-started.md) | 安装、单分子/批量示例 |
| 使用 | [特征化 API](docs/zh/featurization.md) | 函数签名、`coord_mode`、`timeout`、`meta` 状态模型 |
| 使用 | [批量与运行时](docs/zh/batch-and-runtime.md) | 批量语义、并行、参数校验、确定性边界 |
| 使用 | [模型、迁移与 FAQ](docs/zh/model-and-faq.md) | 模型 API、`use_3d` 迁移、常见问题 |
| 使用 | [命令行接口](docs/zh/cli.md) | `unimol-lite` 子命令与参数 |
| 进阶 | [预训练](docs/zh/training.md) | 掩码原子预训练、数据准备、配置、输出 |
| 说明 | [引用与声明](docs/zh/references.md) | 官方关系、论文引用、许可证 |

### 作者与许可

作者：Lvjun Guo、David Niteas

本项目采用 **MIT License**，详见 [LICENSE](LICENSE)；上游 Uni-Mol 同样使用 MIT License。引用信息见下方[引用 / Citation](#citation)。

<a id="en"></a>

## English

UniMol-Lite is a lightweight implementation of [Uni-Mol](https://github.com/deepmodeling/Uni-Mol): it converts SMILES into Uni-Mol model inputs (RDKit conformer generation + tensorization) and provides a forward-model API, masked-atom pretraining and command-line tools.

> This project is an independent third-party implementation and is **not** the official Uni-Mol release. The official implementation, weights and support are available at [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol).

### Installation

Python `>=3.10,<3.12`.

```shell
pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

With an SSH key configured, the SSH URL can also be used:

```shell
pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

Development and pretraining extras:

```shell
pip install -e ".[dev]"
pip install -e ".[train]"
```

### Quick start

Single molecule:

```python
from unimol_lite.data import smiles2unimolitem

item, meta = smiles2unimolitem(
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    coord_mode="auto",
    timeout=30.0,
    return_meta=True,
)
print(meta["status"], meta["coord_mode"])
print({key: tuple(value.shape) for key, value in item.items()})
```

Batch:

```python
from unimol_lite import featurize_batch

results = featurize_batch(["CCO", "not-a-smiles"], workers=4, timeout=30.0)
for item, meta in results:
    shape = None if item is None else tuple(item["src_tokens"].shape)
    print(meta["status"], shape)
```

`item` contains four tensors: `src_tokens`, `src_coord`, `src_distance` and `src_edge_type`. Batch results match the input in length and order, with a `meta` record per row:

| `status` | Meaning |
|---|---|
| `ok` | Success |
| `degraded_2d` | 3D failed or timed out; degraded to 2D |
| `invalid` | SMILES could not be parsed |
| `timeout_error` | Timed out with `on_timeout="error"` |
| `skipped` | Timed out with `on_timeout="skip"` |
| `error` | Unexpected exception caught by the batch layer |

### Model

```python
from unimol_lite.model import UniMolModel

model = UniMolModel.from_pretrained("runs/pretrain/pretrained")
model.eval()
encoder_rep, pair_rep = model(**batch)
# encoder_rep: [B, L, D]
# pair_rep:    [B, L, L, D]
```

### Pretraining and CLI

```shell
# Prepare data
unimol-lite prepare-lmdb --input input.csv --output molecules.lmdb --smiles-column smiles --nproc 8

# Featurize one molecule
unimol-lite featurize --smiles "CCO"

# Masked-atom pretraining
unimol-lite pretrain --lmdb molecules.lmdb --output-dir runs/pretrain --config pretrain.yaml

# Forward/loss self-test
unimol-lite selftest
```

### Documentation

The homepage keeps only installation and a minimal example; API, runtime and pretraining details are expanded in the linked pages. Start with “Getting started”:

| Level | Page | Contents |
|---|---|---|
| Getting started | [Getting started](docs/en/getting-started.md) | Installation, single-molecule and batch examples |
| Usage | [Featurization API](docs/en/featurization.md) | Signatures, `coord_mode`, `timeout`, `meta` status model |
| Usage | [Batch and runtime](docs/en/batch-and-runtime.md) | Batch semantics, parallelism, validation, determinism |
| Usage | [Model, migration and FAQ](docs/en/model-and-faq.md) | Model API, `use_3d` migration, common questions |
| Usage | [CLI](docs/en/cli.md) | `unimol-lite` subcommands and options |
| Advanced | [Pretraining](docs/en/training.md) | Masked-atom pretraining, data preparation, configuration, output |
| Reference | [References and notices](docs/en/references.md) | Official project, citation, license |

### Authors and license

Authors: Lvjun Guo, David Niteas

This project is released under the **MIT License**; see [LICENSE](LICENSE). Upstream Uni-Mol is also MIT-licensed. See [Citation](#citation) below.

<a id="citation"></a>

## 引用 / Citation

使用本项目时，应同时引用 Uni-Mol 原始论文。When using this project, cite the original Uni-Mol paper:

```bibtex
@inproceedings{zhou2023unimol,
  title     = {Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author    = {Gengmo Zhou and Zhifeng Gao and Qiankun Ding and Hang Zheng and Hongteng Xu and Zhewei Wei and Linfeng Zhang and Guolin Ke},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=6K2RM6wVqKu}
}
```

本项目不分发官方预训练权重或受单独条款约束的数据。This project does not distribute official pretrained weights or data covered by separate terms.

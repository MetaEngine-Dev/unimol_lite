# UniMol-Lite

**English** · [简体中文](#zh-cn)

<a id="english"></a>

## English

UniMol-Lite is a lightweight implementation of [Uni-Mol](https://github.com/deepmodeling/Uni-Mol): it turns SMILES into Uni-Mol model inputs (RDKit conformer generation + tensorization) and provides a forward-model API, masked-atom pretraining and a CLI.

> This project is an independent third-party implementation, **not** the official Uni-Mol release. Official implementations, weights and support are available at [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol).

### Docs

- **English:** [Documentation index](docs/en/index.md)
- **简体中文：** [文档索引](docs/zh-CN/index.md)

The docs are ordered by how deep you want to go:

| If you need to... | Read |
|---|---|
| Run something once | [Installation](docs/en/01-installation.md) → [Quickstart](docs/en/02-quickstart.md) |
| Look up featurization | [Featurization API](docs/en/03-featurization.md) |
| Use batch parallelism and timeouts | [Batch and runtime](docs/en/04-batch-and-runtime.md) |
| Load a model or migrate from `use_3d` | [Model, migration and FAQ](docs/en/05-model-and-faq.md) |
| Run masked-atom pretraining | [Pretraining](docs/en/06-training.md) |
| Look up a command | [CLI](docs/en/07-cli.md) |
| Understand why only code is shipped | [Why code-only](docs/en/08-why-code-only.md) |
| Check citation and license | [References and notices](docs/en/09-references.md) |

### Just want the result?

The repository does not ship official Uni-Mol weights or training data. To get model outputs, featurize your molecules and train or export a model locally. If an AI assistant is doing the work, point it at [`AGENTS.md`](./AGENTS.md); it will ask for the data source and license before pretraining.

### What's here

- SMILES → Uni-Mol input tensors (`src_tokens`, `src_coord`, `src_distance`, `src_edge_type`)
- Single-molecule and batch featurization with per-row status
- `UniMolModel` forward API and Hugging Face-style export/load
- Masked-atom pretraining and a CLI (`prepare-lmdb`, `featurize`, `pretrain`, `selftest`)

### No data or model files

- The repository does not ship official Uni-Mol pretrained weights or training data.
- Pretraining data and LMDB files are supplied by the user.
- UniMol-Lite is a modified implementation of Uni-Mol, not the official release.

### Install

Python `>=3.10,<3.12` (3.10 or 3.11) is required.

```bash
python -m pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

For development and pretraining:

```bash
git clone https://github.com/MetaEngine-Dev/unimol_lite.git
cd unimol_lite
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"
```

### Commands

```bash
# Featurize one molecule
unimol-lite featurize --smiles "CCO"

# Prepare an LMDB from a CSV
unimol-lite prepare-lmdb --input input.csv --output molecules.lmdb --smiles-column smiles --nproc 8

# Masked-atom pretraining
unimol-lite pretrain --lmdb molecules.lmdb --output-dir runs/pretrain --config pretrain.yaml

# Forward/loss self-test
unimol-lite selftest
```

### Status

Featurization, batch runtime, pretraining and CLI are covered by tests. Tests use synthetic/small inputs.

### Authors

- Lvjun Guo (Ficere) `<gljx898989@163.com>`
- David Niteas `<david.ql.d@foxmail.com>`

See [`AUTHORS`](./AUTHORS). What's shipped is described in [`NOTICE`](./NOTICE).

### License

MIT. See [`LICENSE`](./LICENSE). Citation metadata is in [`CITATION.cff`](./CITATION.cff); the Uni-Mol paper citation is in [Citation](#citation).

---

<a id="zh-cn"></a>

## 简体中文

[English](#english) · **简体中文**

UniMol-Lite 是 [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) 的轻量实现：把 SMILES 转成 Uni-Mol 模型输入（RDKit 构象生成 + 张量化），并提供前向模型 API、掩码原子预训练与 CLI。

> 本项目是独立第三方实现，**不是** Uni-Mol 官方版本。官方实现、权重与支持见 [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol)。

### 文档

- **English:** [Documentation index](docs/en/index.md)
- **简体中文：** [文档索引](docs/zh-CN/index.md)

文档按使用深度排列：

| 你要做什么 | 看哪里 |
|---|---|
| 先跑通一次 | [安装](docs/zh-CN/01-installation.md) → [快速开始](docs/zh-CN/02-quickstart.md) |
| 查特征化接口 | [特征化 API](docs/zh-CN/03-featurization.md) |
| 使用批量并行与超时 | [批量与运行时](docs/zh-CN/04-batch-and-runtime.md) |
| 加载模型或迁移 `use_3d` | [模型、迁移与 FAQ](docs/zh-CN/05-model-and-faq.md) |
| 跑掩码原子预训练 | [预训练](docs/zh-CN/06-training.md) |
| 查命令参数 | [命令行接口](docs/zh-CN/07-cli.md) |
| 理解为什么只发代码 | [为什么只发代码](docs/zh-CN/08-why-code-only.md) |
| 查引用与许可 | [引用与声明](docs/zh-CN/09-references.md) |

### 只想要结果？

仓库不附带 Uni-Mol 官方权重或训练数据。要得到模型输出，需要先特征化分子，并在本地训练或导出模型。希望 agent 代跑时，把 [`AGENTS.md`](./AGENTS.md) 交给它；它会先与你确认数据来源与许可，再执行预训练。

### 仓库里有什么

- SMILES → Uni-Mol 输入张量（`src_tokens`、`src_coord`、`src_distance`、`src_edge_type`）
- 单分子与批量特征化，逐行状态
- `UniMolModel` 前向 API 与 Hugging Face 风格导出/加载
- 掩码原子预训练与 CLI（`prepare-lmdb`、`featurize`、`pretrain`、`selftest`）

### 没有数据或模型文件

- 仓库不附带 Uni-Mol 官方预训练权重或训练数据。
- 预训练数据与 LMDB 文件由用户提供。
- UniMol-Lite 是 Uni-Mol 的修改版实现，不是官方版本。

### 安装

需要 Python `>=3.10,<3.12`（3.10 或 3.11）。

```bash
python -m pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

开发与预训练依赖：

```bash
git clone https://github.com/MetaEngine-Dev/unimol_lite.git
cd unimol_lite
python -m pip install -e ".[dev]"
python -m pip install -e ".[train]"
```

### 常用命令

```bash
# 单分子特征化
unimol-lite featurize --smiles "CCO"

# 从 CSV 准备 LMDB
unimol-lite prepare-lmdb --input input.csv --output molecules.lmdb --smiles-column smiles --nproc 8

# 掩码原子预训练
unimol-lite pretrain --lmdb molecules.lmdb --output-dir runs/pretrain --config pretrain.yaml

# 前向/损失自检
unimol-lite selftest
```

### 状态

特征化、批量运行时、预训练与 CLI 有测试覆盖；测试使用合成/小规模输入。

### 作者

- Lvjun Guo (Ficere) `<gljx898989@163.com>`
- David Niteas `<david.ql.d@foxmail.com>`

见 [`AUTHORS`](./AUTHORS)。仓库包含什么见 [`NOTICE`](./NOTICE)。

### 许可证

MIT，见 [`LICENSE`](./LICENSE)。引用元数据见 [`CITATION.cff`](./CITATION.cff)；Uni-Mol 论文引用见[引用](#citation)。

---

<a id="citation"></a>

## 引用 / Citation

使用本项目或派生工作时，应同时引用 Uni-Mol 原始论文。When using this project or derivative work, cite the original Uni-Mol paper:

```bibtex
@inproceedings{zhou2023unimol,
  title     = {Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author    = {Gengmo Zhou and Zhifeng Gao and Qiankun Ding and Hang Zheng and Hongteng Xu and Zhewei Wei and Linfeng Zhang and Guolin Ke},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=6K2RM6wVqKu}
}
```

- OpenReview: <https://openreview.net/forum?id=6K2RM6wVqKu>
- Paper: *Uni-Mol: A Universal 3D Molecular Representation Learning Framework*
- Venue: ICLR 2023
- Full notices: [English](docs/en/09-references.md) · [简体中文](docs/zh-CN/09-references.md)

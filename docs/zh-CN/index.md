# UniMol-Lite 文档（简体中文）

**语言：** [English](../en/index.md) · 简体中文

UniMol-Lite 是 [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) 的轻量实现：把 SMILES 转成 Uni-Mol 模型输入，提供前向模型 API、掩码原子预训练与 CLI。它是独立的纯代码项目，不附带官方权重或训练数据。

## 从哪开始

| 你想做什么 | 阅读顺序 |
|---|---|
| 理解为什么只发代码 | [为什么只发代码](./08-why-code-only.md) |
| 安装并跑通一个小例子 | [安装](./01-installation.md) → [快速开始](./02-quickstart.md) |
| 查特征化签名与状态模型 | [特征化 API](./03-featurization.md) |
| 使用批量并行与超时 | [批量与运行时](./04-batch-and-runtime.md) |
| 加载模型或迁移 `use_3d` | [模型、迁移与 FAQ](./05-model-and-faq.md) |
| 跑掩码原子预训练 | [预训练](./06-training.md) |
| 查命令参数 | [命令行接口](./07-cli.md) |
| 查官方关系、引用与许可 | [引用与声明](./09-references.md) |

## 怎么读这些文档

- **第一次用：** [安装](./01-installation.md)，然后跑 [快速开始](./02-quickstart.md)。
- **需要特征化分子？** [特征化 API](./03-featurization.md) → [批量与运行时](./04-batch-and-runtime.md)。
- **需要模型？** 在本地训练或导出；仓库不附带官方权重。见[预训练](./06-training.md)。
- **需要命令？** [命令行接口](./07-cli.md)。

## 你需要自备

- 需要特征化的 SMILES；预训练时还需要自己的 3D 构象数据或 LMDB。
- 需要加载的模型目录：本仓库不附带官方权重，按[预训练](./06-training.md)训练并导出，或使用已有的导出目录。
- 如需官方 Uni-Mol 权重，请从上游获取。

作者见 [`AUTHORS`](../../AUTHORS)，仓库包含什么见 [`NOTICE`](../../NOTICE)。

## 作者与许可

- Lvjun Guo (Ficere) `<gljx898989@163.com>`
- David Niteas `<david.ql.d@foxmail.com>`
- 许可证：MIT（仅代码）

**下一页：** [安装](./01-installation.md)

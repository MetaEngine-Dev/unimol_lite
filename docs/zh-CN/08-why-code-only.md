# 为什么只发代码

**中文** | [English](../en/08-why-code-only.md) · [返回主页](../../README.md#zh-cn)

**层级：** 0 — 范围

**前置条件：** 无。

本仓库只包含代码、配置和文档，不包含训练数据，也不包含 Uni-Mol 官方预训练权重。

## 权重

Uni-Mol 官方预训练权重由上游项目分发，受其来源页面条款约束，本仓库不转存。需要官方权重时，请前往 [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol)。

本仓库提供掩码原子预训练流程，训练后会在 `runs/pretrain/pretrained/` 导出 Hugging Face 风格的推理目录，见[预训练](./06-training.md)。

## 数据

仓库不附带构象数据或 LMDB 文件。特征化只需要 SMILES；预训练需要用户自己的 3D 构象数据或 LMDB。

## 与本项目的关系

UniMol-Lite 是 Uni-Mol 的轻量修改版实现，不是官方版本。引用与许可见[引用与声明](./09-references.md)。

**下一页：** [引用与声明](./09-references.md)

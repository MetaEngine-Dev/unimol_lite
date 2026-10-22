# 引用与声明

**中文** | [English](../en/09-references.md) · [返回主页](../../README.md#zh-cn)

**层级：** 3 — 参考

**前置条件：** [为什么只发代码](./08-why-code-only.md)。

## 本项目与官方 Uni-Mol 的关系

`unimol_lite` 是独立的第三方实现，**不是** Uni-Mol 的官方版本，也不由 Uni-Mol 作者维护或背书。它只保留了一条轻量的 SMILES 特征化、模型前向与掩码原子预训练通路，方便在本地数据上做实验和二次开发。

官方实现、预训练权重、下游工具与官方支持见以下仓库：

- 官方仓库：<https://github.com/deepmodeling/Uni-Mol>
- 官方工具包：<https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_tools>

## 论文引用

使用本项目或派生工作时，应同时引用 Uni-Mol 原始论文：

```bibtex
@inproceedings{zhou2023unimol,
  title     = {Uni-Mol: A Universal 3D Molecular Representation Learning Framework},
  author    = {Gengmo Zhou and Zhifeng Gao and Qiankun Ding and Hang Zheng and Hongteng Xu and Zhewei Wei and Linfeng Zhang and Guolin Ke},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=6K2RM6wVqKu}
}
```

- OpenReview：<https://openreview.net/forum?id=6K2RM6wVqKu>
- 论文标题：*Uni-Mol: A Universal 3D Molecular Representation Learning Framework*
- 发表：ICLR 2023

## 许可证

- 本项目 `unimol_lite` 以 **MIT License** 发布，完整文本见仓库根目录的 [`LICENSE`](../../LICENSE)。
- 上游 [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) 同样使用 MIT License；本项目的部分实现与接口参考了上游代码与文档。
- 本项目不分发 Uni-Mol 官方预训练权重或其他受单独条款约束的数据。另行下载权重或数据时，应遵守其来源页面的许可与使用条款。

**下一页：** [文档索引](./index.md)

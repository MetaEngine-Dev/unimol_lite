# 解决方案四层结构（L1–L4）

[English](../en/10-solution-layers.md) | **简体中文** · [文档索引](./index.md)

**级别：** 1 — 概念

**前置：** [安装](./01-installation.md)。

**下一页：** [特征化 API](./03-featurization.md) · [数据容器](./11-dataspec.md)

## 为什么要分层

UniMol-Lite 是一个**解决方案**，不只是一个模型：它同时提供网络、预训练配方、标准调用面与最短可用路径。把这四层分开，标准调用面才有意义——绕过标准调用直接取内部编码器的消费方，会悄悄继承另一套语义（不同的默认值、不同的坐标模式、不同的后处理），而没有任何报错。下面的分层规则让这类旁路可见，并在评审时可拒绝。

## 四层

| 层 | 定义什么 | 入口 | 边界 |
| --- | --- | --- | --- |
| **L1 架构** | 网络及其配置 | `unimol_lite.model`：`UniMolModel`、`UniMolConfig`、`UniMolTokenizer`；编码器部件在 `unimol_lite.model.layers`（内部） | 只定义前向；不做数据准备、不做训练循环、不做 I/O |
| **L2 训练** | 掩码原子预训练 | `unimol_lite.training`：`UniMolPretrainModule`、`LMDBPretrainDataset` / `UniMolPretrainDataset`、`mask_unimol_item`、`collate_pretrain_batch`；CLI `unimol-lite prepare-lmdb` / `pretrain` / `selftest` | 产出模型导出；不产出检索向量。需要 `train` 可选依赖 |
| **L3 标准调用** | 应用使用的契约 | `unimol_lite.api`：`featurize`、`featurize_batch`、`build_model`、`load_model`、`forward_features`；数据 I/O 在 `unimol_lite.dataspec` | **应用只应调用这一层**。输出为分子特征（pre-alignment） |
| **L4 最简使用** | 最短端到端路径 | [`examples/minimal_features.py`](../../examples/minimal_features.py) 与下方代码块 | 只走 L3；不新增自身逻辑 |

### 依赖方向（单向）

```text
L4 最简使用 ─┐
             ├─> L3 标准调用 ─> L1 架构
L2 训练    ──┘                  （L2 同样只碰 L1）
应用层 ───────> 只走 L3
```

- L3 不伸手进 `unimol_lite.training`；L4 不伸手进 L1 / L2。
- 应用层（对齐/检索包、服务、GUI）只调 L3，不调更下层。若某用例看起来需要内部编码器，正确的修法是在 L3 补入口——`forward_features` 就是为此存在。

## 职责边界：分子特征（pre-alignment）

`unimol_lite` 产出的是**分子特征**：原子级编码器状态 `(batch, atoms + 2, dim)` 与对状态 `(batch, L, L, heads)`。

它**刻意不做**：

- 施加 LayerNorm 把分子向量变成可与谱图比较的向量——那是消费方解决方案包的跨模态对齐步骤（CLIP 式模型里就是 CLS 行上的 `mol_ln`）；
- 嵌入谱图、做谱图归一化或谱图侧任何处理；
- 持有向量库、度量选择或检索入口。

理由：可检索向量必须由**唯一一个**实现产出，而那个实现必须同时持有两种模态。对齐层负责 LN 与度量契约；本库负责特征化与编码器。

## L3 契约

### 特征化

| 调用 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `featurize(smiles, ...)` | 单个 SMILES | 张量字典；`return_meta=True` 时为 `(item \| None, meta)` | `return_meta=False` 且无法特征化时抛 `ValueError` |
| `featurize_batch(smiles, ...)` | SMILES 可迭代对象 | `list[(item \| None, meta)]`，与输入等长保序 | 单行失败不会中断整批 |

单个分子的张量（`L = 重原子数 + 2`，含 CLS/SEP 两个 token）：

| 键 | 形状 | dtype | 单位 |
| --- | --- | --- | --- |
| `src_tokens` | `(L,)` | int64 | token id；token 化前会去掉氢 |
| `src_coord` | `(L, 3)` | float32 | 埃（Å），已按坐标均值居中；CLS/SEP 行为 0 |
| `src_distance` | `(L, L)` | float32 | 埃，成对欧氏距离 |
| `src_edge_type` | `(L, L)` | int64 | `token_i * 词表大小 + token_j` |

两个入口共享的参数与默认值：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `coord_mode` | `"auto"` | `"auto"` → 400 原子以内走 3D，否则 2D；`"3d"`；`"2d"` |
| `timeout` | `30.0` 秒 | 单个构象的墙钟预算（embed + 力场）；`None` 关闭 |
| `on_timeout` | `"fallback_2d"` | `"fallback_2d"` / `"error"` / `"skip"` |
| `rng_seed` | `42` | 构象选择与原子截断的种子 |
| `max_atoms` | `256` | 原子上限，超出按种子随机截断 |
| `workers` | `min(8, CPU)` | 仅批量；`1` = 串行 |

`meta` 给出 `status`（`ok` / `degraded_2d` / `invalid` / `timeout_error` / `skipped` / `error`）、`coord_mode`、`reason` 与 `elapsed_s`。对 3D 训练的模型，`degraded_2d` 应视为分布外输入。

底层的 `unimol_lite.data.smiles2unimolitem` 保留历史默认值（`timeout=None`）以兼容旧调用；标准入口是 `unimol_lite.api.featurize`，其预算与批量默认一致。

### 模型

| 调用 | 输入 | 输出 |
| --- | --- | --- |
| `build_model(config=None, device=None)` | 配置字典或 `UniMolConfig` | **随机权重**、推理态模型（结构/演示用途） |
| `load_model(path, device=None)` | 预训练导出的目录（`config.json` + 权重） | 推理态模型 |
| `forward_features(model, items, device=None)` | 单个 item 或 item 列表 | `(encoder_rep, pair_rep)`：`(B, L, D)` 与 `(B, L, L, heads)`，float32，无梯度 |

`forward_features` 会把 item 列表填充到 8 的倍数（`L` 为填充后长度），并拒绝训练态模型——训练态返回的是损失头，不是对表示。

### 数据容器

读写 `SMS-DataSpec` 的 parquet-block 容器属于 L3：见[数据容器](./11-dataspec.md)。

## L2 说明：预训练数据保持上游兼容

预训练 LMDB 布局跟随上游 Uni-Mol 项目，**豁免**于容器协议：它是内部训练产物，不是交换格式，改动会破坏与上游工具的兼容性。容器协议覆盖的是分子/谱图数据集（见[数据容器](./11-dataspec.md)）。

## L4：最短可用路径

```python
from unimol_lite import build_model, featurize, forward_features

# 仓库不附带权重：这是随机初始化的模型，即结构演示
model = build_model({
    "encoder_layers": 1, "encoder_embed_dim": 32, "encoder_ffn_embed_dim": 64,
    "encoder_attention_heads": 4, "max_seq_len": 32,
})
item = featurize("CCO")                      # 模型输入张量
encoder_rep, pair_rep = forward_features(model, item)
print(encoder_rep.shape, pair_rep.shape)     # (1, L, 32) (1, L, L, 4)
```

有训练/导出好的模型时，把 `build_model(...)` 换成 `load_model("runs/pretrain/pretrained")`。`encoder_rep[:, 0, :]`（CLS 行）是惯用的分子级向量；把它变成可检索向量是对齐层的职责，不是本库的。

同一条路径可直接跑脚本：

```bash
python examples/minimal_features.py                                  # 随机权重
python examples/minimal_features.py --model-dir runs/pretrain/pretrained --smiles CCO c1ccccc1O
```

## 兼容性

既有导入路径不变：`unimol_lite.data.smiles2unimolitem`、`unimol_lite.data.featurize_batch`、`unimol_lite.batch.featurize_batch`、`unimol_lite.model.UniMolModel`。`unimol_lite.__all__` 按层分组；`unimol_lite.dataspec` 保持为子模块，因为它需要可选的 `dataspec` 依赖。

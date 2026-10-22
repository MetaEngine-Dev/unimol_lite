# 模型、迁移与 FAQ

**中文** | [English](../en/05-model-and-faq.md) · [返回主页](../../README.md#zh-cn)

**层级：** 3 — 参考

**前置条件：** [快速开始](./02-quickstart.md)。

## 模型 API

```python
# model 为 UniMolModel 实例
encoder_rep, pair_rep = model(**batch)
# encoder_rep: [B, L, D]    原子/分子级表示
# pair_rep:    [B, L, L, D] 原子对级表示
```

- `batch` 需要包含 `src_tokens` / `src_distance` / `src_coord` / `src_edge_type`；单个分子（无 batch 维）会自动补一维。
- 当 `config.mode == "infer"` 时前向返回 `(encoder_rep, pair_rep)`；训练模式（`mode != "infer"`）返回包含 loss 相关项的元组。
- 常用做法是取 `encoder_rep[:, 0, :]`（`[CLS]` 位置）作为整个分子的表征，用于性质预测、相似度计算等下游任务。

## use_3d 迁移

`use_3d` 仅作为兼容参数保留，使用时发出 `DeprecationWarning`。

| 旧写法 | 新写法 |
| --- | --- |
| `smiles2unimolitem(smi, use_3d=False)` | `smiles2unimolitem(smi, coord_mode="auto")` |
| `smiles2unimolitem(smi, use_3d=True)` | `smiles2unimolitem(smi, coord_mode="3d")` |

同时传入 `coord_mode` 时以 `coord_mode` 为准，`use_3d` 被忽略。

旧默认 `use_3d=True` 对所有分子使用 3D 坐标；`coord_mode="auto"` 对超过 400 个原子的分子使用 2D。需要超大分子的 3D 结果时，显式传入 `coord_mode="3d"`。

批量处理可使用 `featurize_batch(smiles_list, workers=8, timeout=30.0)`，返回结果与输入等长保序，并逐项给出状态。

## FAQ

### `degraded_2d` 的成因

3D 嵌入失败或超出 `timeout` 预算时，默认的 `on_timeout="fallback_2d"` 将坐标降级为 2D。长链脂类等柔性强、原子多的分子较易触发。`meta["reason"]` 可区分 `timeout`、`embed_failed` 与 `mmff_failed`。

### 2D 结果的使用

主模型按 3D 构象训练，2D 属于训练分布外输入。下游应按 `meta["status"]` 过滤 `degraded_2d`，或对这类结果降权。

### 同一分子两次结果不一致

- 检查 `timeout`：启用预算时，不同机器或负载下可能一次 3D、一次 2D。需要严格可复现时使用 `timeout=None`。
- 检查输入参数：`coord_mode`、`rng_seed`、`max_atoms`、`coord_pool_count` 是否一致。
- 检查 RDKit 版本：坐标与构象绑定 RDKit 版本（已验证 `2025.09.1`），升级可能改变结果。

### 升级 RDKit 的影响

RDKit 版本变化可能改变构象生成与优化结果，进而改变张量。下游依赖固定特征时，应锁定版本并重新生成已有数据。

### 批量结果的坏行过滤

```python
good = [(item, meta) for item, meta in results if item is not None and meta["status"] == "ok"]
```

需要同时保留降级 2D 时，额外接受 `meta["status"] == "degraded_2d"`。

### 与外部调度器集成

`smiles2unimolitem` 可以放入外部进程池，但通常没有必要：`featurize_batch` 已内置线程池，并处理行对齐与逐项状态。

### 线程数

默认 `workers=min(8, cpu)`。收益取决于分子大小与硬件，增加线程不一定更快。

## 开发与测试

```shell
# 可编辑安装 + 开发依赖
pip install -e ".[dev]"

# 全量测试
pytest tests/ -q

# 格式化与静态检查
black .
isort .
flake8 .
```

**下一页：** [预训练](./06-training.md)

# 模型、迁移与 FAQ

**中文** | [English](../en/model-and-faq.md)

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

`use_3d` 已弃用，仍可用但会发 `DeprecationWarning`：

| 旧写法 | 新写法 |
| --- | --- |
| `smiles2unimolitem(smi, use_3d=False)` | `smiles2unimolitem(smi, coord_mode="auto")` |
| `smiles2unimolitem(smi, use_3d=True)` | `smiles2unimolitem(smi, coord_mode="3d")` |

同时显式传 `coord_mode` 时以 `coord_mode` 为准（不再看 `use_3d`）。

下游唯一现有调用点 `uni_classify/model_tools/system.py:41` 的对照：

```python
# 迁移前
unimolitem = smiles2unimolitem(smiles, use_3d=False)

# 迁移后（行为一致：≤400 原子 3D，>400 原子 2D）
unimolitem = smiles2unimolitem(smiles, coord_mode="auto")
```

> 注意：旧默认 `use_3d=True` 对所有尺寸走 3D，而 `coord_mode="auto"` 对 >400 原子走 2D。若原来依赖超大分子的 3D 结果，请显式写 `coord_mode="3d"`。

若希望把调用点的 dask 进程池换成库内线程池，可用 `featurize_batch(smiles_list, workers=8, timeout=30.0)`（等长保序、逐项状态）。

## FAQ

### 为什么我的分子变成了 degraded_2d？

3D 嵌入失败或超出 `timeout` 预算时，`on_timeout="fallback_2d"`（默认）会降级成 2D 坐标。长链脂类等柔性强、原子多的分子容易嵌入失败，这属于预期行为。可查看 `meta["reason"]` 区分 `timeout` / `embed_failed` / `mmff_failed`。

### 2D 结果能当成普通结果用吗？

要谨慎。主模型按 3D 构象训练，2D 属于训练分布外输入；不要静默把 `degraded_2d` 当正常结果，应按 `meta["status"]` 过滤或降权。

### 同一分子两次结果不一样？

- 检查 `timeout`：启用预算时，不同机器/负载下可能一次 3D、一次 2D。需要可复现时用 `timeout=None`。
- 检查输入参数：`coord_mode`、`rng_seed`、`max_atoms`、`coord_pool_count` 是否一致。
- 检查 RDKit 版本：坐标与构象绑定 RDKit 版本（当前验证 `2025.09.1`），升级可能改变结果。

### 升级 RDKit 会影响已有数据吗？

会。RDKit 版本变化可能改变构象生成与优化结果，进而改变张量。若下游依赖固定特征，请锁定版本并重跑已有数据。

### 批量结果里怎么过滤坏行？

```python
good = [(item, meta) for item, meta in results if item is not None and meta["status"] == "ok"]
```

需要把降级 2D 也保留时，额外接受 `meta["status"] == "degraded_2d"`。

### 能用自己的进程池包 `smiles2unimolitem` 吗？

可以，但通常没有必要：`featurize_batch` 已经内置线程池并处理了行对齐与状态。

### 应该开多少线程？

从 `workers=8` 开始；收益取决于分子大小与机器，线程更多不一定更快。

## 开发与测试

```shell
# 可编辑安装 + 开发依赖
pip install -e ".[dev]"

# 全量测试
pytest tests/ -q
# 89 passed, 1 skipped

# 格式化与静态检查
black .
isort .
flake8 .
```

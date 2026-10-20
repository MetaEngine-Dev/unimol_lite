# 特征化 API

**中文** | [English](../en/featurization.md)

## 总览

特征化的入口都在 `unimol_lite.data`（`featurize_batch` 同时从 `unimol_lite` 顶层导出）：

| 函数 | 作用 |
| --- | --- |
| `smiles2coords` | SMILES → 原子符号 + 坐标列表 |
| `smiles2unimolitem` | SMILES → 模型输入字典（含状态 `meta`） |
| `generate_unimolitem` | 原子 + 坐标 → 模型输入字典 |
| `featurize_batch` | 批量并行特征化，逐行返回状态 |
| `FeaturizationTimeout` | 超时异常（仅在特定参数组合下抛出） |

## smiles2coords

```python
smiles2coords(
    smiles,
    count=10,
    coord_mode="auto",
    return_meta=False,
    use_3d=_UNSET,
    *,
    timeout=None,
    on_timeout="fallback_2d",
)
```

- 返回 `(atoms, coordinates_list)`；`count` 是生成的 3D 构象数，默认 `10`，因此坐标列表长度为 `count + 1`（10 个 3D + 1 个 2D）。
- `return_meta=True` 时返回 `(result, meta)`。
- 非法 SMILES：`return_meta=True` 返回 `(None, meta)` 且 `status="invalid"`；`return_meta=False` 返回 `[None, None]`。
- 失败/超时的处理由 `on_timeout` 决定（见下文）。

## smiles2unimolitem

```python
smiles2unimolitem(
    smiles,
    tokenizer=UniMolTokenizer(),
    coord_pool_count=1,
    random_seed=42,
    max_atoms=256,
    normalize_coord=True,
    use_3d=_UNSET,
    *addl_seeds,
    coord_mode=_UNSET,
    rng_seed=None,
    return_meta=False,
    timeout=None,
    on_timeout="fallback_2d",
)
```

返回模型输入字典：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `src_tokens` | `[L]` | 原子 token 序列，含 `[CLS]` / `[SEP]`，氢原子已去除 |
| `src_coord` | `[L, 3]` | 坐标，首尾为补零的 `[CLS]` / `[SEP]` 行 |
| `src_distance` | `[L, L]` | 原子间距离矩阵 |
| `src_edge_type` | `[L, L]` | 原子对边类型 |

参数要点：

- `coord_pool_count` 生成多个构象后按 `random_seed` 采样一个；`rng_seed` 可单独指定随机种子（默认取 `random_seed`）。
- `max_atoms` 为截断上限，超限时用局部 RNG 随机保留 `max_atoms` 个原子。
- `normalize_coord=True` 时坐标按质心平移。
- `use_3d` 已弃用，见[迁移说明](model-and-faq.md)。
- `return_meta=True` 返回 `(item | None, meta)`；默认 `False` 只返回 `item`。

返回值约定（`return_meta=False`）：

- 正常：返回 `item`。
- 非法 SMILES：抛 `TypeError`。
- `on_timeout="error"` 且超时：抛 `FeaturizationTimeout`。
- `on_timeout="skip"` 且超时：返回 `None`。
- `on_timeout="fallback_2d"`：返回降级后的 2D `item`（无 meta 时不报错）。

## generate_unimolitem

```python
generate_unimolitem(
    atoms,
    coordinates,
    tokenizer=UniMolTokenizer(),
    random_seed=42,
    max_atoms=256,
    normalize_coord=True,
    *addl_seeds,
)
```

当已有原子符号与坐标时可直接调用：去除氢原子、按 `max_atoms` 截断、按需平移归一化，再生成与 `smiles2unimolitem` 相同的四个张量。它不做构象生成，因此不需要 RDKit 嵌入，也不涉及 `timeout`。

## featurize_batch

```python
featurize_batch(
    smiles,
    *,
    workers=None,
    timeout=30.0,
    on_timeout="fallback_2d",
    coord_mode="auto",
    rng_seed=42,
    max_atoms=256,
    tokenizer=None,
    window=None,
)
```

返回 `list[(item | None, meta)]`，与输入等长保序。语义、参数校验与并行细节见[批量与性能](batch-and-performance.md)。

## coord_mode

| 值 | 行为 |
| --- | --- |
| `auto`（默认） | ≤400 原子走 3D，>400 原子走 2D |
| `3d` | 强制 3D（所有尺寸）；失败/超时按 `on_timeout` 处理 |
| `2d` | 强制 2D，不调用 `EmbedMolecule`，也没有构象生成开销 |

`auto` 的原子数按加氢后的分子统计，与旧 `use_3d=False` 的推理路径等价。

## timeout 与 on_timeout

`timeout` 是**每个构象**的墙钟预算，覆盖 RDKit 嵌入（embed）与 MMFF 优化：

- `None` 关闭预算，不设超时；单分子 API 默认 `None`，批量层默认 `30.0` 秒。
- RDKit 原生超时只接受整数秒（亚秒向上取 1 秒），精确预算由 embed/MMFF 结束后的墙钟检查兜底。
- `count > 1`（`smiles2coords`）时总预算为 `count * timeout`，任一构象失败即丢弃全部 3D 结果。
- 失败不重试：相同参数 + 相同种子结果确定，重试只会翻倍耗时。

`on_timeout` 决定超时（或 3D 嵌入失败）后的动作：

| 值 | 行为 | `status` | `item` |
| --- | --- | --- | --- |
| `fallback_2d`（默认） | 降级 2D | `degraded_2d` | dict |
| `error` | 报错（`return_meta=True` 时记状态） | `timeout_error` | `None` |
| `skip` | 跳过 | `skipped` | `None` |

## meta 状态模型

`meta` 是字典，包含 `status` / `coord_mode` / `reason` / `elapsed_s`：

| `status` | 含义 | `item` |
| --- | --- | --- |
| `ok` | 正常（3D 或 2D 模式） | dict |
| `degraded_2d` | 3D 失败或超时，已降级 2D | dict |
| `timeout_error` | 超时，且 `on_timeout="error"` | `None` |
| `invalid` | SMILES 无法解析或结构处理失败 | `None` |
| `skipped` | 超时，且 `on_timeout="skip"` | `None` |
| `error` | 批量层捕获的未预期异常 | `None` |

- `coord_mode`：实际使用的模式（降级后为 `"2d"`）；SMILES 无法解析时为 `None`（若调用方显式传了 `"3d"` / `"2d"`，则保留该显式值）。注意它是**实际模式**，不是调用方传入的模式。
- `reason`：`timeout` / `embed_failed` / `mmff_failed` / `parse_failed` / 异常类型名。
- `elapsed_s`：本次调用耗时（秒）。

## FeaturizationTimeout

`FeaturizationTimeout` 继承 `RuntimeError`，只在 `on_timeout="error"` 且 `return_meta=False` 时抛出；若 `return_meta=True`，同类情况返回 `(None, meta)` 并把 `status` 记为 `timeout_error`。批量层永远逐行记状态，不会因单行超时抛出。

## 确定性

随机性全部来自每次调用创建的局部 `np.random.RandomState`，不读写全局 `np.random` 状态；同一输入 + 同一种子在单线程/多线程下结果一致。结果绑定 RDKit 版本（当前验证版本 `2025.09.1`）；启用 `timeout` 时，同一分子在不同机器或不同负载下可能 3D 成功、也可能降级 2D。需要可复现输出时用 `timeout=None`，或按 `meta["status"]` 过滤降级行。

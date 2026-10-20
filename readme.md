# UniMol-Lite

UniMol-Lite 是 [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) 的简化版本实现。

## 文档 / Documentation

- [文档导航](docs/README.md)（中英双语）
- 中文：[概览](docs/zh/index.md) · [快速开始](docs/zh/getting-started.md) · [特征化 API](docs/zh/featurization.md) · [批量与性能](docs/zh/batch-and-performance.md) · [模型、迁移与 FAQ](docs/zh/model-and-faq.md)
- English: [Overview](docs/en/index.md) · [Getting started](docs/en/getting-started.md) · [Featurization API](docs/en/featurization.md) · [Batch and performance](docs/en/batch-and-performance.md) · [Model, migration and FAQ](docs/en/model-and-faq.md)

## 安装

```shell
# 安装 tag v0.1.1
pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.1.1"

# 开发环境设置（可编辑安装 + 开发依赖）
pip install -e ".[dev]"

# 运行测试
pytest tests/ -q

# 运行代码格式化
black .
isort .
flake8 .
```

> 仓库目前为私有，公开后以上 HTTPS 地址可直接使用；维护者访问私有仓库时可改用 SSH：`pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git@v0.1.1"`。

## 使用示例

### 1. 准备数据

#### 方式一: 直接使用 SMILES

```python
from unimol_lite.data import smiles2unimolitem

# 转换单个SMILES
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
item = smiles2unimolitem(smiles)
```

#### 方式二: 使用 LMDB 数据集

1. 将 SMILES 转换为 LMDB:

```python
import pandas as pd

# 读取CSV文件中的SMILES
df = pd.read_csv("examples/example_prepare_lmdb.csv")
```

```shell
# 使用脚本批量转换
python examples/prepare_lmdb.py -i input.csv -o output.lmdb -c SMILES_COLUMN
```

2. 使用 LMDBDataset 加载数据:

```python
from unimol_lite.data import LMDBDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = LMDBDataset("output.lmdb")

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=dataset.collater
)
```

#### 输入数据格式
- SMILES: 标准SMILES格式的分子字符串
- CSV文件: 包含SMILES列的CSV文件,可通过`-c`参数指定列名

#### 中间数据格式
LMDB数据集中每个条目包含:
```python
{
    "smiles": str,                # 原始SMILES
    "item": {
        "src_tokens": tensor,     # [L] 原子类型的token序列
        "src_coord": tensor,      # [L,3] 原子3D坐标
        "src_distance": tensor,   # [L,L] 原子间距离矩阵
        "src_edge_type": tensor,  # [L,L] 原子间边类型
    }
}
```

#### 批处理数据格式
DataLoader输出的batch包含:
```python
{
    "src_tokens": tensor,     # [B,L] 
    "src_coord": tensor,      # [B,L,3]
    "src_distance": tensor,   # [B,L,L]
    "src_edge_type": tensor,  # [B,L,L]
}
```
其中B为批大小,L为序列长度(包含padding)。

### 2. 分子采样

#### 单分子

```python
from unimol_lite.data import smiles2unimolitem

item, meta = smiles2unimolitem("CCO", coord_mode="auto", timeout=30.0, return_meta=True)
# item: 模型输入 dict（src_tokens / src_coord / src_distance / src_edge_type）
# meta: 状态模型，见下文；本行 meta["status"] == "ok"
```

`return_meta=False`（默认）时只返回 `item`；非法 SMILES 且无 `meta` 时抛 `TypeError`，`on_timeout="error"` 超时则抛 `FeaturizationTimeout`。

#### 批量

```python
from unimol_lite import featurize_batch

smiles_list = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "not-a-smiles"]
results = featurize_batch(smiles_list, workers=8, timeout=30.0)
# 等长保序；每项 (item | None, meta)
for item, meta in results:
    print(meta["status"], meta["elapsed_s"])
```

- `workers` 默认 `min(8, os.cpu_count())`；使用线程池并行特征化；
- 单个坏分子不中断整批（逐项记 `status`），重复 SMILES 不折叠；非法批量参数（`coord_mode` / `on_timeout` / `timeout` / `workers` / `window`）在提交前抛 `ValueError`。

#### `coord_mode`

| 值 | 行为 |
|---|---|
| `auto`（默认） | ≤400 原子走 3D，>400 原子走 2D（等价旧 `use_3d=False` 的推理路径） |
| `3d` | 强制 3D（所有尺寸）；失败/超时按 `on_timeout` 处理 |
| `2d` | 强制 2D，不做构象生成（不调用 `EmbedMolecule`） |

#### `on_timeout`

| 值 | 行为 |
|---|---|
| `fallback_2d`（默认） | 降级 2D，`meta["status"] == "degraded_2d"` |
| `error` | `item` 为 `None`（`return_meta=False` 时抛 `FeaturizationTimeout`），状态 `timeout_error` |
| `skip` | `item` 为 `None`，状态 `skipped` |

`timeout` 是**每个构象**的墙钟预算（覆盖 embed + MMFF）；`count > 1` 时总预算为 `count * timeout`，任一构象失败即丢弃全部 3D 结果。`timeout=None`（单分子 API 默认）不设预算，批量层默认 `timeout=30.0`。RDKit 原生 timeout 粒度为整数秒（亚秒按 1s 执行），精确预算由 embed/MMFF 结束后的墙钟检查兜底。

#### `meta` 状态模型

| `status` | 含义 | `item` |
|---|---|---|
| `ok` | 正常（3D 或 2D 模式） | dict |
| `degraded_2d` | 3D 失败/超时，已降级 2D | dict |
| `timeout_error` | 超时，且 `on_timeout="error"` | `None` |
| `invalid` | SMILES 无法解析 | `None` |
| `skipped` | 超时，且 `on_timeout="skip"` | `None` |
| `error` | 批量层捕获的未预期异常 | `None` |

`meta` 另含 `coord_mode`（实际使用的模式）、`reason`（`timeout` / `embed_failed` / `mmff_failed` / `parse_failed` / 异常类型名）与 `elapsed_s`（秒）。

> **2D 降级是训练分布外输入**：主模型按 3D 构象训练。下游不要静默把 `degraded_2d` 当正常结果使用，应按 `meta["status"]` 过滤或降权。

#### 迁移：`use_3d` → `coord_mode`

| 旧写法 | 新写法 |
|---|---|
| `smiles2unimolitem(smi, use_3d=False)` | `smiles2unimolitem(smi, coord_mode="auto")` |
| `smiles2unimolitem(smi, use_3d=True)` | `smiles2unimolitem(smi, coord_mode="3d")` |

`use_3d` 仍可用，但会发 `DeprecationWarning`；同时显式传 `coord_mode` 时以 `coord_mode` 为准。

下游唯一现有调用点 `uni_classify/model_tools/system.py:41` 的对照：

```python
# 迁移前
unimolitem = smiles2unimolitem(smiles, use_3d=False)

# 迁移后（行为一致：≤400 原子 3D，>400 原子 2D）
unimolitem = smiles2unimolitem(smiles, coord_mode="auto")
```

> 注意：旧默认 `use_3d=True` 对所有尺寸走 3D，而 `coord_mode="auto"` 对 >400 原子走 2D——若原来依赖超大分子的 3D 结果，请显式用 `coord_mode="3d"`。
> 若希望把调用点的 dask 进程池换成库内线程池，可用 `featurize_batch(smiles_list, workers=8, timeout=30.0)`（等长保序，逐项状态）。

### 3. 输出说明

模型输出包含两部分:
1. encoder_output: 分子/原子级别的表示,形状为 [batch_size, seq_len, hidden_dim]
2. pair_output: 原子对级别的表示,形状为 [batch_size, seq_len, seq_len, hidden_dim]

这些表示可用于:
- 分子性质预测
- 分子相似度计算
- 分子结构分析
- 等下游任务

## 性能与确定性

在 32 vCPU 机器上，125 个代表性分子（重原子 5–80）的批量特征化：

| workers | 时间 | 加速 |
|---|---|---|
| 1 | 27.82 s | 1.00× |
| 4 | 8.73 s | 3.19× |
| 8 | 5.56 s | **5.00×** |

- 默认 `workers=min(8, cpu)`；继续增加线程收益递减；
- 大分子可能因 3D 构象生成失败而自动降级 2D（`meta["status"] == "degraded_2d"`）。

确定性：

- 同一输入与种子在串行/多线程下结果一致；
- 结果依赖 RDKit 版本（验证版本 `2025.09.1`）；
- 启用 `timeout` 时，慢机器上可能降级 2D；需要严格可复现时用 `timeout=None`，或按 `meta["status"]` 过滤。

基准脚本：`benchmarks/bench_featurize.py`。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

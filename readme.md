# UniMol-Lite

UniMol-Lite 是 [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) 的简化版本实现。

## 安装

```shell
# 使用 uv 安装（推荐；当前为私有仓库，需要该仓库的 SSH 访问权限）
uv pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git"

# 或者使用 pip 安装
pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git"

# 开发环境设置
uv pip install -e ".[dev]"  # 安装开发依赖
# 或者使用pip
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 运行代码格式化
black .
isort .
flake8 .
```

> **PyPI 发布：TBD**（尚未发布，当前只能从上面的 git 地址安装）。

## 预训练模型

- **Featurization（SMILES → 模型输入）不需要任何权重**：`smiles2unimolitem` / `featurize_batch` 只依赖 RDKit，安装后即可使用。
- **只有 `UniMolModel.from_pretrained(...)` 需要权重**，且权重需用户自备：把 `base_unimol/` 放到本地，再将目录路径传给 `UniMolModel.from_pretrained("base_unimol")`。
- 权重下载入口 **TBD**（原内部 bohrium 链接仅内网可达，已暂从本文移除）。

权重文件组织如下:

```
base_unimol/
|-- config.json
`-- model.safetensors
```

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

- `workers` 默认 `min(8, os.cpu_count())`；RDKit 的 embed/MMFF 释放 GIL，**线程池优于进程池**；
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

`timeout` 是**每个构象**的墙钟预算（覆盖 embed + MMFF）；`count > 1` 时总预算为 `count * timeout`，任一构象失败即丢弃全部 3D 结果。`timeout=None`（单分子 API 默认）走历史无预算路径，批量层默认 `timeout=30.0`。RDKit 原生 timeout 粒度为整数秒（亚秒按 1s 执行），精确预算由 embed/MMFF 结束后的墙钟检查兜底。

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

### 3. 使用模型

```python
from unimol_lite.model import UniMolModel, UniMolConfig

# 初始化配置
config = UniMolConfig()

# 加载预训练模型（需自备 base_unimol/ 权重目录）
model = UniMolModel.from_pretrained("base_unimol")

# 使用模型进行推理
encoder_output, pair_output = model(**batch)
# encoder_output: [B,L,D] 分子表示
# pair_output: [B,L,L,D] 原子对表示
# 一般而言，你可能倾向于取encoder_rep[:, 0, :]作为整个SMILES的表征
```

### 4. 输出说明

模型输出包含两部分:
1. encoder_output: 分子/原子级别的表示,形状为 [batch_size, seq_len, hidden_dim]
2. pair_output: 原子对级别的表示,形状为 [batch_size, seq_len, seq_len, hidden_dim]

这些表示可用于:
- 分子性质预测
- 分子相似度计算
- 分子结构分析
- 等下游任务

## 性能与确定性

125 个 HMDB 分层样本（重原子 5 桶 × 25，`benchmarks/bench_featurize.py --per-bin 25`，32 vCPU）：

| workers | 墙钟 | 加速比 |
|---|---|---|
| 1 | 27.82 s | 1.00× |
| 4 | 8.73 s | 3.19× |
| 8 | 5.56 s | **5.00×** |

- 8 线程为甜点，16 线程效率降到约 52%，故默认 `workers=min(8, cpu)`；
- 该样本中 **33/125 条长链脂类 `embed_failed → 2D`**，与改造前逐位比对一致（非本库引入的回归）；2D 路径的张量化受 GIL 限制，是加速比止步 5× 的主因。

确定性：

- 随机性全部来自**局部 `np.random.RandomState`**（不再读写全局 `np.random` 状态），同一输入 + 同一种子在单线程/多线程下逐位一致；
- 结果**绑定 RDKit 版本**（当前验证版本 **2025.09.1**），升级 RDKit 可能改变构象与坐标；
- **启用 `timeout` 时，同一分子在不同机器（或不同负载）上可能 3D 成功、也可能降级 2D**。需要可复现输出时用 `timeout=None`，或按 `meta["status"]` 过滤降级行。

复跑基准：

```shell
python benchmarks/bench_featurize.py --csv <HMDB csv> --per-bin 25 --workers 1,4,8
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

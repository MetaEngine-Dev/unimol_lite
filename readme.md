# UniMol-Lite

UniMol-Lite 是 [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) 的简化版本实现。

## 安装

```shell
pip install ssh://git@github.com/macroversus/unimol_lite.git
# Download pre-trained models
wget http://ohgy1146835.bohrium.tech:50001/download/base_unimol.zip
unzip base_unimol.zip
```

## 预训练模型

预训练模型文件组织如下:

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

### 3. 使用模型

```python
from unimol_lite.model import UniMolModel, UniMolConfig

# 初始化配置
config = UniMolConfig()

# 加载预训练模型
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

## 替代下载方式

如果wget不可用,也可以使用以下方式下载:

```bash
# 使用curl
curl -L http://ohgy1146835.bohrium.tech:50001/download/base_unimol.zip -o base_unimol.zip

# 使用aria2
aria2c http://ohgy1146835.bohrium.tech:50001/download/base_unimol.zip

# 断点续传
wget -c http://ohgy1146835.bohrium.tech:50001/download/base_unimol.zip
curl -L -C - http://ohgy1146835.bohrium.tech:50001/download/base_unimol.zip -o base_unimol.zip
aria2c -c http://ohgy1146835.bohrium.tech:50001/download/base_unimol.zip
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

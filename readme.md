# UniMol Lite

这是一个轻量级的UniMol模型，用于快速的SMILES表征

## 安装

```shell
# 安装依赖
git clone https://github.com/macroversus/unimol_lite.git && pip install ./unimol_lite
```

## 使用

### 输入SMILES获得表征（推荐在Inference时使用）

```python
from unimol_lite.model import UniMolModel
from unimol_lite.data import smiles2unimolitem

model = UniMolModel.from_pretrained("unimol_lite/base_unimol") # 替换成你repo中的base_unimol目录路径
test_smiles = "Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1"
item = smiles2unimolitem(smiles=test_smiles)
encoder_rep, encoder_pair_rep = model(**item)
```

### 输入预先生成的atoms和coordinates获得表征（推荐在Uni-Mol微调中使用）

```python
from unimol_lite.model import UniMolModel
from unimol_lite.data import generate_unimolitem, smiles2coords

model = UniMolModel.from_pretrained("unimol_lite/base_unimol") # 替换成你repo中的base_unimol目录路径
test_smiles = "Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1"
atoms, coordinates_list = smiles2coords(test_smiles, 1) # 生成Atoms和coordinate
coordinates = coordinates_list[0] # 选择其中一个构象
item = generate_unimolitem(atoms=atoms, coordinates=coordinates)
encoder_rep, encoder_pair_rep = model(**item)

```

### 输出说明

***encoder_rep***: (batch_size, atoms_count, hidden_size)

***encoder_rep***: (batch_size, atoms_count, atoms_count， hidden_size)

上述atoms_count: 不包括H, 第一个为cls，最后一个为sep

一般而言，你可能倾向于取encoder_rep[:, 0, :]作为整个SMILES的表征

## Dataset样例

本项目提供了两个简单的Dataset样例，供您参考使用

### BaseDataset

```python
from unimol_lite.data import BaseDataset
from torch.utils.data import DataLoader

base_dataset = BaseDataset(
    smiles_lists: list = [...], # SMILES列表
)
# 可以使用dataset自带的collater对数据进行对齐处理
base_dataloader = DataLoader(base_dataset, batch_size=1, shuffle=False, collate_fn=base_dataset.collater)
```

### LMDBDataset

本项目提供了一个预先生成LMDB数据的样例examples/prepare_lmdb.py，以及对应的输入文件examples/example_prepare_lmdb.csv

```python
from unimol_lite.data import LMDBDataset
from torch.utils.data import DataLoader

#lmdb中的数据格式为：
# {
#     "lmdb_key": {
#         "item": {...}, # 必须包含——利用unimol_lite.data.generate_unimolitem产生的item
#         "other_keys_you_need": {...}，
#         ...
#     }
# }
lmdb_dataset = LMDBDataset(
    lmdb_path: str = "path/to/your/lmdb", # LMDB路径
    included_keys: list = [b"0", b"1", ...], # 用于此dataset的lmdb中的keys
)

lmdb_dataloader = DataLoader(lmdb_dataset, batch_size=1, shuffle=False, collate_fn=lmdb_dataset.collater)
```

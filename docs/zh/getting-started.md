# 快速开始

**中文** | [English](../en/getting-started.md)

## 环境要求

- Python `>=3.10,<3.12`
- 运行依赖（安装时自动拉取）：`rdkit>=2025.9,<2026`、`transformers>=4.47,<5`、`numpy<2`、`pandas>=2.0`、`scipy>=1.10`、`lmdb>=1.4`、`torch>=2.0`

## 安装

```shell
pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.1.1"
```

仓库目前为私有，公开后上述 HTTPS 地址可直接使用；维护者访问私有仓库时可改用 SSH 地址：

```shell
pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git@v0.1.1"
```

开发环境（可编辑安装 + 开发依赖）：

```shell
pip install -e ".[dev]"
```

## 最小示例：单个分子

```python
from unimol_lite.data import smiles2unimolitem

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
item, meta = smiles2unimolitem(smiles, coord_mode="auto", timeout=30.0, return_meta=True)

print(meta["status"], meta["coord_mode"], round(meta["elapsed_s"], 3))
print({key: tuple(value.shape) for key, value in item.items()})
```

`item` 是模型输入字典，四个张量分别对应原子 token、坐标、距离矩阵与边类型；`meta` 记录本次特征化的状态，详见[特征化 API](featurization.md)。

## 最小示例：批量

```python
from unimol_lite import featurize_batch

smiles_list = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "not-a-smiles"]
results = featurize_batch(smiles_list, workers=4, timeout=30.0)

for smi, (item, meta) in zip(smiles_list, results):
    shape = None if item is None else tuple(item["src_tokens"].shape)
    print(smi, "->", meta["status"], shape)
```

批量结果与输入**等长保序**：非法 SMILES 只影响自己那一行（`status="invalid"`），不会中断整批。语义细节见[批量与性能](batch-and-performance.md)。

## 运行测试

```shell
pytest tests/ -q
# 89 passed, 1 skipped
```


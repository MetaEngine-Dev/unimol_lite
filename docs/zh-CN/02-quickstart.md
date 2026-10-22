# 快速开始

**中文** | [English](../en/02-quickstart.md) · [返回主页](../../README.md#zh-cn)

**层级：** 1 — 开始

**前置条件：** [安装](./01-installation.md)。

## 最小示例：单个分子

```python
from unimol_lite.data import smiles2unimolitem

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
item, meta = smiles2unimolitem(smiles, coord_mode="auto", timeout=30.0, return_meta=True)

print(meta["status"], meta["coord_mode"], round(meta["elapsed_s"], 3))
print({key: tuple(value.shape) for key, value in item.items()})
```

`item` 是模型输入字典，四个张量分别对应原子 token、坐标、距离矩阵与边类型；`meta` 记录本次特征化的状态，详见[特征化 API](03-featurization.md)。

## 最小示例：批量

```python
from unimol_lite import featurize_batch

smiles_list = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "not-a-smiles"]
results = featurize_batch(smiles_list, workers=4, timeout=30.0)

for smi, (item, meta) in zip(smiles_list, results):
    shape = None if item is None else tuple(item["src_tokens"].shape)
    print(smi, "->", meta["status"], shape)
```

批量结果与输入**等长保序**：非法 SMILES 只影响该行（`status="invalid"`），不会中断整批。语义细节见[批量与运行时](04-batch-and-runtime.md)。

## 运行测试

```shell
pytest tests/ -q
```

## 下一步

- 全部函数签名与状态模型：[特征化 API](03-featurization.md)
- 批量并行与运行时：[批量与运行时](04-batch-and-runtime.md)
- 模型加载与迁移：[模型、迁移与 FAQ](05-model-and-faq.md)

**下一页：** [特征化 API](./03-featurization.md)

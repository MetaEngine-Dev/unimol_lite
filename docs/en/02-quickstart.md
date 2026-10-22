# Getting started

[中文](../zh-CN/02-quickstart.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 1 — Start

**Prerequisites:** [Installation](./01-installation.md).

## Minimal example: one molecule

```python
from unimol_lite.data import smiles2unimolitem

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
item, meta = smiles2unimolitem(smiles, coord_mode="auto", timeout=30.0, return_meta=True)

print(meta["status"], meta["coord_mode"], round(meta["elapsed_s"], 3))
print({key: tuple(value.shape) for key, value in item.items()})
```

`item` is the model input dict: four tensors for atom tokens, coordinates, the distance matrix and edge types. `meta` reports how this featurization went; see the [featurization API](03-featurization.md).

## Minimal example: a batch

```python
from unimol_lite import featurize_batch

smiles_list = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "not-a-smiles"]
results = featurize_batch(smiles_list, workers=4, timeout=30.0)

for smi, (item, meta) in zip(smiles_list, results):
    shape = None if item is None else tuple(item["src_tokens"].shape)
    print(smi, "->", meta["status"], shape)
```

Results are **the same length as the input and in input order**: an invalid SMILES only spoils its own row (`status="invalid"`) and never aborts the batch. See [batch and runtime](04-batch-and-runtime.md) for the details.

## Running the tests

```shell
pytest tests/ -q
```

## Next steps

- Full signatures and the status model: [Featurization API](03-featurization.md)
- Batch parallelism and runtime: [Batch and runtime](04-batch-and-runtime.md)
- Model loading and migration: [Model, migration and FAQ](05-model-and-faq.md)

**Next:** [Featurization API](./03-featurization.md)

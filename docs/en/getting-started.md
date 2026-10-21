# Getting started

[中文](../zh/getting-started.md) | **English**

## Requirements

- Python `>=3.10,<3.12`
- Runtime dependencies (installed automatically): `rdkit>=2025.9,<2026`, `transformers>=4.47,<5`, `numpy<2`, `pandas>=2.0`, `scipy>=1.10`, `lmdb>=1.4`, `torch>=2.0`

## Installation

```shell
pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

Maintainers can use SSH:

```shell
pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

For development (editable install plus dev extras):

```shell
pip install -e ".[dev]"
```

## Minimal example: one molecule

```python
from unimol_lite.data import smiles2unimolitem

smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
item, meta = smiles2unimolitem(smiles, coord_mode="auto", timeout=30.0, return_meta=True)

print(meta["status"], meta["coord_mode"], round(meta["elapsed_s"], 3))
print({key: tuple(value.shape) for key, value in item.items()})
```

`item` is the model input dict: four tensors for atom tokens, coordinates, the distance matrix and edge types. `meta` reports how this featurization went; see the [featurization API](featurization.md).

## Minimal example: a batch

```python
from unimol_lite import featurize_batch

smiles_list = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "not-a-smiles"]
results = featurize_batch(smiles_list, workers=4, timeout=30.0)

for smi, (item, meta) in zip(smiles_list, results):
    shape = None if item is None else tuple(item["src_tokens"].shape)
    print(smi, "->", meta["status"], shape)
```

Results are **the same length as the input and in input order**: an invalid SMILES only spoils its own row (`status="invalid"`) and never aborts the batch. See [batch and runtime](batch-and-runtime.md) for the details.

## Running the tests

```shell
pytest tests/ -q
# 94 passed, 1 skipped
```


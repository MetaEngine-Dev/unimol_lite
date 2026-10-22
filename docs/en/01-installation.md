# Installation

[中文](../zh-CN/01-installation.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 1 — Start

**Prerequisites:** Python `>=3.10,<3.12` and `pip`.

## Install from Git

```shell
python -m pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

With an SSH key configured, the SSH URL can also be used:

```shell
python -m pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

## Development install

```shell
git clone https://github.com/MetaEngine-Dev/unimol_lite.git
cd unimol_lite
python -m pip install -e ".[dev]"
```

## Pretraining dependencies

```shell
python -m pip install -e ".[train]"
```

The `train` extra provides PyTorch Lightning, PyYAML and safetensors.

## Verify the install

```shell
python -c "import unimol_lite; print(unimol_lite.__version__)"
unimol-lite --help
unimol-lite selftest
```

`selftest` featurizes `CCO` and runs a tiny forward/loss check; it does not train or write weights.

## Build a wheel

```shell
python -m pip install build
python -m build
python -m pip install dist/unimol_lite-*.whl
```

## Common installation problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: pytorch_lightning` | Install `.[train]` |
| `ModuleNotFoundError: rdkit` | Check the Python version and network, then reinstall dependencies |
| Python 3.12 or newer | Use Python 3.10 or 3.11 |
| Build failure | Upgrade `pip` and confirm `hatchling` is available |

**Next:** [Quickstart](./02-quickstart.md)

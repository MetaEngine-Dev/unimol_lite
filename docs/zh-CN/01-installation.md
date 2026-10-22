# 安装

**中文** | [English](../en/01-installation.md) · [返回主页](../../README.md#zh-cn)

**层级：** 1 — 开始

**前置条件：** Python `>=3.10,<3.12`（3.10 或 3.11）与 `pip`。

## 从 Git 安装

```shell
python -m pip install "git+https://github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

已配置 SSH 密钥时，也可使用：

```shell
python -m pip install "git+ssh://git@github.com/MetaEngine-Dev/unimol_lite.git@v0.2.1"
```

`smiles2class` 以相同的 Python 范围依赖本库；组合安装时请使用 Python 3.10 或 3.11。

## 开发安装

```shell
git clone https://github.com/MetaEngine-Dev/unimol_lite.git
cd unimol_lite
python -m pip install -e ".[dev]"
```

## 预训练依赖

```shell
python -m pip install -e ".[train]"
```

`train` extra 提供 PyTorch Lightning、PyYAML 与 safetensors。

## 验证安装

```shell
python -c "import unimol_lite; print(unimol_lite.__version__)"
unimol-lite --help
unimol-lite selftest
```

`selftest` 会特征化 `CCO`，跑一次微型前向与损失检查，不训练、不写权重。

## 构建 wheel

```shell
python -m pip install build
python -m build
python -m pip install dist/unimol_lite-*.whl
```

## 常见安装问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: pytorch_lightning` | 安装 `.[train]` |
| `ModuleNotFoundError: rdkit` | 检查 Python 版本与网络，重新安装依赖 |
| Python 3.12 及以上 | 使用 Python 3.10 或 3.11 |
| 构建失败 | 升级 `pip`，确认 `hatchling` 可用 |

**下一页：** [快速开始](./02-quickstart.md)

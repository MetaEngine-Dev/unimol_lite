# 命令行接口

**中文** | [English](../en/07-cli.md) · [返回主页](../../README.md#zh-cn)

**层级：** 3 — 参考

**前置条件：** [快速开始](./02-quickstart.md)。

安装后可直接使用 `unimol-lite` 命令；未安装时也可以用 `python -m unimol_lite.cli`。

```shell
unimol-lite --help
unimol-lite prepare-lmdb --help
unimol-lite pretrain --help
unimol-lite selftest --help
```

## 命令总览

| 命令 | 作用 | 是否需要 `[train]` 依赖 |
| --- | --- | --- |
| `prepare-lmdb` | 把 CSV 中的 SMILES 特征化成 LMDB | 否 |
| `featurize` | 特征化单个 SMILES，打印形状或保存张量 | 否 |
| `pretrain` | 用 PyTorch Lightning 做掩码原子预训练 | 是 |
| `selftest` | 特征化 + 微型模型前向/损失自检 | 否 |

## prepare-lmdb

```shell
unimol-lite prepare-lmdb \
  --input molecules.csv \
  --output molecules.lmdb \
  --smiles-column smiles \
  --nproc 8 \
  --coord-mode auto \
  --timeout 30 \
  --max-atoms 256
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input`, `-i` | 必填 | 输入 CSV |
| `--output`, `-o` | 必填 | 输出 LMDB 文件 |
| `--smiles-column` | `CANON_SMILES` | SMILES 列名 |
| `--nproc` | `1` | 特征化进程数；`1` 为串行 |
| `--chunk-size` | `1000` | 多进程任务的块大小参考值 |
| `--coord-mode` | `auto` | `auto` / `3d` / `2d` |
| `--timeout` | `30.0` | 每个构象的秒级预算 |
| `--max-atoms` | `256` | 单分子最大原子数 |
| `--keep-errors` | 关闭 | 保留失败行（`item` 为空，附 `error`） |
| `--quiet` | 关闭 | 不打印汇总 JSON |

成功时以退出码 `0` 结束，并输出：

```json
{"written": 1234, "skipped": 5, "total": 1239, "path": "molecules.lmdb"}
```

## featurize

```shell
unimol-lite featurize --smiles "CCO"
unimol-lite featurize --smiles "CCO" --output cco.npz
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--smiles` | 必填 | 输入 SMILES |
| `--output`, `-o` | 无 | 可选输出文件；`.npz` 或 `.pt` / `.pth` |
| `--coord-mode` | `auto` | `auto` / `3d` / `2d` |
| `--timeout` | `30.0` | 每个构象的秒级预算 |
| `--max-atoms` | `256` | 最大原子数 |
| `--quiet` | 关闭 | 保存文件时不打印路径 |

不传 `--output` 时打印 JSON，包含 `status`、`coord_mode` 和四个张量的形状；传 `--output` 时保存 `src_tokens` / `src_coord` / `src_distance` / `src_edge_type` 及 `meta`。

## pretrain

```shell
unimol-lite pretrain \
  --lmdb molecules.lmdb \
  --output-dir runs/pretrain \
  --config pretrain.yaml
```

| 参数 | 说明 |
| --- | --- |
| `--lmdb` | 训练 LMDB，必填 |
| `--val-lmdb` | 验证 LMDB；省略时从训练集按 90/10 划分 |
| `--output-dir` | 输出目录，必填 |
| `--config` | YAML 配置 |
| `--max-epochs` | 最大 epoch 数 |
| `--max-steps` | 最大优化步数；`-1` 表示按 epoch |
| `--batch-size` | 批大小 |
| `--lr` | 学习率 |
| `--weight-decay` | AdamW 权重衰减 |
| `--warmup-ratio` | 线性 warmup 占总步数的比例 |
| `--num-workers` | DataLoader worker 数 |
| `--device` | `cpu`、`cuda` 或 Lightning accelerator |
| `--precision` | 例如 `32-true`、`16-mixed` |
| `--seed` | 随机种子 |
| `--resume` | 从 checkpoint 继续 |
| `--quiet` | 不打印结束提示 |

命令行参数优先于 YAML 中的同名项。训练细节、配置字段和输出结构见[预训练](06-training.md)。

## selftest

```shell
unimol-lite selftest
unimol-lite selftest --device cuda
```

自检会：

1. 用 RDKit 特征化 `CCO`；
2. 构造一个 1 层、32 维的临时 `UniMolModel`；
3. 计算掩码 token / 坐标 / 距离损失；
4. 输出各损失值，并检查均为有限数。

它不训练、不写文件，适合安装后确认依赖与模型前向可用。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功 |
| `2` | 参数错误、文件不存在、依赖缺失或运行时错误 |

错误信息写到 stderr，例如：

```text
unimol-lite: error: pretraining requires pytorch-lightning; install with '.[train]'
```

**下一页：** [为什么只发代码](./08-why-code-only.md)

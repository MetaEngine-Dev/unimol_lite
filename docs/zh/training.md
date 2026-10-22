# 预训练

**中文** | [English](../en/training.md) · [返回主页](../../README.md#zh)

本页说明如何使用 `unimol-lite` 在本地 3D 构象数据上做 Uni-Mol 风格的掩码原子预训练。实现参考了 [Uni-Mol 官方预训练代码](https://github.com/deepmodeling/Uni-Mol)，但不是官方实现，也不分发官方预训练权重。

## 预训练目标

Uni-Mol 的预训练是一个掩码原子建模任务。每个分子先被转换成 Uni-Mol 输入张量，然后在原子 token、坐标和原子对距离三个层面加入掩码/扰动：

| 目标 | 输入 | 监督信号 | 默认权重 |
| --- | --- | --- | --- |
| 掩码 token | 被选中原子的 token 被替换为 `[MASK]`、随机原子或保持不变 | 原始原子 token | `1.0` |
| 掩码坐标 | 被选中原子的坐标加入噪声 | 原始坐标 | `5.0` |
| 掩码距离 | 噪声坐标计算出的原子对距离 | 原始坐标的原子对距离 | `10.0` |

另外保留 Uni-Mol 的 `x_norm` 与 pair 表示范数正则项，默认权重均为 `0.01`。这些权重均可在配置文件中调整；默认值是一组通用起点，应按数据分布调整。

## 1. 准备数据

### 输入 CSV

CSV 至少包含一列 SMILES。默认列名是 `CANON_SMILES`，也可以通过 `--smiles-column` 指定：

```csv
smiles
CCO
c1ccccc1
CC(=O)OC1=CC=CC=C1C(=O)O
```

### 生成 LMDB

```shell
unimol-lite prepare-lmdb \
  --input molecules.csv \
  --output molecules.lmdb \
  --smiles-column smiles \
  --nproc 8 \
  --coord-mode auto \
  --timeout 30
```

- `--coord-mode auto|3d|2d`：与 `smiles2unimolitem` 的含义一致；`auto` 对大分子可能使用 2D 坐标。
- `--timeout`：每个构象的墙钟预算。
- 解析失败或构象生成失败的分子默认**跳过**；加 `--keep-errors` 可以把错误行写入 LMDB，预训练数据集读取时会跳过空条目。
- LMDB 中每个条目保存 `{"smiles": str, "item": {...}}`，`item` 的结构与[特征化 API](featurization.md)一致。
- `--nproc 1` 使用串行模式，便于调试；多进程模式的进度条按行汇总。

## 2. 配置

预训练由一份 YAML 配置驱动。命令行参数会覆盖配置文件中的同名项：

```yaml
model:
  encoder_layers: 15
  encoder_embed_dim: 512
  encoder_ffn_embed_dim: 2048
  encoder_attention_heads: 64
  max_seq_len: 512
  dropout: 0.1
  attention_dropout: 0.1
  emb_dropout: 0.1

data:
  batch_size: 16
  num_workers: 4
  mask_prob: 0.15
  leave_unmasked_prob: 0.05
  random_token_prob: 0.05
  noise_type: uniform
  noise: 1.0

optim:
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  warmup_ratio: 0.06
  masked_token_weight: 1.0
  masked_coord_weight: 5.0
  masked_dist_weight: 10.0
  x_norm_weight: 0.01
  delta_pair_weight: 0.01

trainer:
  max_epochs: 10
  max_steps: -1
  accelerator: auto
  precision: 32-true
  seed: 1
```

### 掩码参数

| 参数 | 含义 |
| --- | --- |
| `mask_prob` | 每个原子被选中的概率；若一个分子都没选中，会强制选一个 |
| `random_token_prob` | 选中原子替换为随机原子 token 的概率 |
| `leave_unmasked_prob` | 选中原子保持原 token 的概率 |
| 其余选中原子 | 替换为 `[MASK]` |
| `noise_type` | `uniform`、`normal`、`trunc_normal` 或 `none` |
| `noise` | 坐标噪声幅度；`uniform` 时每维在 `[-noise, noise]` 采样；只作用于被选中且未保留原 token 的原子 |

> 默认 `0.05 + 0.05` 意味着约 90% 的选中原子使用 `[MASK]`，而不是 BERT 常见的 80/10/10。需要不同比例时直接改配置。

## 3. 启动预训练

```shell
unimol-lite pretrain \
  --lmdb molecules.lmdb \
  --output-dir runs/pretrain \
  --config pretrain.yaml
```

常用覆盖项：

```shell
unimol-lite pretrain \
  --lmdb molecules.lmdb \
  --val-lmdb validation.lmdb \
  --output-dir runs/pretrain \
  --config pretrain.yaml \
  --max-epochs 20 \
  --batch-size 8 \
  --lr 5e-5 \
  --device cuda \
  --precision 16-mixed \
  --num-workers 4
```

- 不传 `--val-lmdb` 时，按 90/10 从训练 LMDB 中划分训练/验证集；只有 1 条数据时会报错。
- `--max-steps` 用于快速冒烟；`--resume` 可以接 Lightning checkpoint。
- `--device` 传给 Lightning 的 `accelerator`；CPU 用 `cpu`，单卡用 `cuda`，其他后端按 Lightning 支持填写。
- 训练需要可选的 `train` 依赖：

```shell
pip install -e ".[train]"
```

## 4. 输出

```text
runs/pretrain/
├── checkpoints/
│   ├── epoch=000-val_loss=....ckpt
│   └── last.ckpt
├── last.ckpt
└── pretrained/
    ├── config.json
    └── model.safetensors
```

`pretrained/` 是 Hugging Face 风格的推理导出：`config.json` 中的 `mode` 会被写成 `infer`，可以直接用 `UniMolModel.from_pretrained("runs/pretrain/pretrained")` 加载并返回 `(encoder_rep, pair_rep)`。

```python
from unimol_lite.model import UniMolModel

model = UniMolModel.from_pretrained("runs/pretrain/pretrained")
model.eval()
encoder_rep, pair_rep = model(**batch)
```

## 5. 自检

不下载数据也可以先跑内置的微型前向/损失自检：

```shell
unimol-lite selftest
```

它会用 `CCO` 做一次特征化，构造一个 1 层、32 维的临时模型，计算三个预训练损失并检查有限性；不会训练、不会写权重。

## 6. 实现边界

- 本实现聚焦「掩码原子 + 坐标/距离恢复」这一条预训练主线，不包含官方仓库里所有数据增广、2D conformer 组合和分布式训练配置。
- 训练数据质量、构象生成设置和 RDKit 版本都会影响结果；换数据或升级 RDKit 后应重新生成 LMDB。
- 本项目不是 Uni-Mol 官方版本，也不分发官方预训练权重。官方实现与权重见[官方仓库](https://github.com/deepmodeling/Uni-Mol)。
- 引用与许可证信息见[引用与声明](references.md)。

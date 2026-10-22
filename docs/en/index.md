# UniMol-Lite documentation (English)

**Language:** English · [简体中文](../zh-CN/index.md)

UniMol-Lite is a lightweight implementation of [Uni-Mol](https://github.com/deepmodeling/Uni-Mol): it turns SMILES into Uni-Mol model inputs, provides a forward-model API, masked-atom pretraining and a CLI. It is an independent, code-only project: it does not ship official weights or training data.

## Pick a starting point

| If you want to... | Read |
|---|---|
| Understand why only code is shipped | [Why code-only](./08-why-code-only.md) |
| Install and run a small example | [Installation](./01-installation.md) → [Quickstart](./02-quickstart.md) |
| Look up featurization signatures and status | [Featurization API](./03-featurization.md) |
| Use batch parallelism and timeouts | [Batch and runtime](./04-batch-and-runtime.md) |
| Load a model or migrate from `use_3d` | [Model, migration and FAQ](./05-model-and-faq.md) |
| Run masked-atom pretraining | [Pretraining](./06-training.md) |
| Look up a command | [CLI](./07-cli.md) |
| Check official relationship, citation and license | [References and notices](./09-references.md) |

## How to read these docs

- **First time here?** [Install](./01-installation.md), then run the [Quickstart](./02-quickstart.md).
- **Need to featurize molecules?** [Featurization API](./03-featurization.md) → [Batch and runtime](./04-batch-and-runtime.md).
- **Need a model?** Train or export one locally; no official weights are shipped. See [Pretraining](./06-training.md).
- **Need a command?** [CLI](./07-cli.md).

## What you need to provide

- SMILES to featurize; for pretraining, your own 3D conformer data or an LMDB.
- A model directory to load: no official weights are shipped; train and export one with [Pretraining](./06-training.md), or use an existing export.
- For official Uni-Mol weights, obtain them upstream.

Authors are in [`AUTHORS`](../../AUTHORS). What is shipped is in [`NOTICE`](../../NOTICE).

## Authors and license

- Lvjun Guo (Ficere) `<gljx898989@163.com>`
- David Niteas `<david.ql.d@foxmail.com>`
- License: MIT (code only)

**Next:** [Installation](./01-installation.md)

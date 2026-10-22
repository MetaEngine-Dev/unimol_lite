# Why code-only

[中文](../zh-CN/08-why-code-only.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 0 — Scope

**Prerequisites:** None.

This repository contains code, configuration and documentation only. It does not include training data or official Uni-Mol pretrained weights.

## Weights

Official Uni-Mol pretrained weights are distributed by the upstream project under its own terms; this repository does not redistribute them. For official weights, use [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol).

This repository provides a masked-atom pretraining pipeline. After training it exports a Hugging Face-style inference directory under `runs/pretrain/pretrained/`; see [Pretraining](./06-training.md).

## Data

The repository ships no conformer data or LMDB files. Featurization only needs SMILES; pretraining needs your own 3D conformer data or LMDB.

## Relationship to this project

UniMol-Lite is a lightweight, modified implementation of Uni-Mol, not the official release. See [References and notices](./09-references.md) for citation and license information.

**Next:** [References and notices](./09-references.md)

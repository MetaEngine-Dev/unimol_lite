# UniMol-Lite overview

[中文](../zh/index.md) | **English**

A lightweight implementation of [Uni-Mol](https://github.com/dptech-corp/Uni-Mol): a complete path from SMILES to model inputs (RDKit conformer generation + tensorization), plus a forward-model API.

## What this is

UniMol-Lite splits "SMILES → model input" into two layers:

- **Featurization** (`smiles2unimolitem` / `featurize_batch`): RDKit generates and MMFF-optimizes 3D conformers, then converts them into the four tensors `src_tokens` / `src_coord` / `src_distance` / `src_edge_type`.
- **Model** (`UniMolModel`): encoder/pair representations matching Uni-Mol.

## Features

- **Featurization out of the box**: RDKit is the only chemistry dependency.
- **Controllable coordinate mode**: `coord_mode="auto" | "3d" | "2d"`; `auto` uses 3D up to 400 atoms and 2D above that.
- **Predictable timeouts**: one wall-clock budget per conformer, with `on_timeout` deciding between 2D degradation, an error, or a skip.
- **Traceable status**: `return_meta=True` gives you `status` / `coord_mode` / `reason` / `elapsed_s`.
- **Parallel batches**: `featurize_batch` uses a thread pool, preserves input order and length, keeps duplicate SMILES, and isolates per-row failures.
- **Determinism**: randomness comes from a local `np.random.RandomState`; results are tied to the RDKit version.

## Documentation map

| Page | What you will find |
| --- | --- |
| [Getting started](getting-started.md) | Installation, minimal runnable examples |
| [Featurization API](featurization.md) | Every signature, `coord_mode`, `timeout`, `meta` |
| [Batch and performance](batch-and-performance.md) | Batch semantics, benchmark numbers, determinism boundaries |
| [Model, migration and FAQ](model-and-faq.md) | Model API, `use_3d` migration, common questions, development and tests |

## Version and license

Current version `0.1.0` (tag `v0.1.0`), installed via git. Licensed under MIT. The code derives from [Uni-Mol](https://github.com/dptech-corp/Uni-Mol) (DP Technology, MIT).

# Batch and performance

[中文](../zh/batch-and-performance.md) | **English**

## Batch semantics

`featurize_batch` is a parallel wrapper around `smiles2unimolitem` with these guarantees:

- **Equal length, input order**: one result per input, in the original order.
- **Duplicates are kept**: repeated SMILES each get their own row (no folding).
- **Per-row isolation**: an unexpected exception on one row becomes `status="error"` for that row and the batch continues.
- **Bounded submission**: a sliding window caps the number of in-flight futures instead of submitting everything at once; the input and result lists still hold `N` entries each.
- **`workers=1` is a plain serial loop**: no thread pool, no scheduling overhead.
- **Up-front validation**: invalid `coord_mode` / `on_timeout` / `timeout` / `workers` / `window` raise `ValueError` before any work is submitted (an empty input is validated too).

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `workers` | `min(8, cpu)` | thread count; `1` runs serially; `<= 0` raises `ValueError` |
| `timeout` | `30.0` | per-conformer wall-clock budget (embed + MMFF); `None` disables it |
| `on_timeout` | `"fallback_2d"` | `"fallback_2d"` / `"error"` / `"skip"` |
| `coord_mode` | `"auto"` | forwarded per row to `smiles2unimolitem` |
| `rng_seed` | `42` | seed for conformer sampling and atom truncation |
| `max_atoms` | `256` | atom cap |
| `tokenizer` | `None` | shared tokenizer (read-only); defaults to a shared instance |
| `window` | `max(workers * 4, 16)` | in-flight window size |

## Timeouts and errors

- The batch layer always calls the single-molecule API with `return_meta=True`, so a timeout under `on_timeout="error"` shows up as `status="timeout_error"` with `item=None` on that row. It **never aborts the batch**, and you never need to catch `FeaturizationTimeout`.
- `on_timeout="skip"` yields `status="skipped"`; `fallback_2d` yields `status="degraded_2d"` and still returns an `item`.
- Invalid SMILES yield `status="invalid"`; unexpected exceptions yield `status="error"` with the exception type as `reason`.
- Per-row isolation means **the result list can contain rows with `item=None`**: downstream code must filter on `meta["status"]`.

## Benchmarks

125 stratified HMDB samples (5 heavy-atom bins × 25, 32 vCPU):

| workers | wall clock | speed-up |
| --- | --- | --- |
| 1 | 27.82 s | 1.00× |
| 4 | 8.73 s | 3.19× |
| 8 | 5.56 s | **5.00×** |

- **8 threads is the sweet spot**: 16 threads drop to ~52% efficiency, hence the `workers=min(8, cpu)` default.
- In this sample **33/125 long-chain lipids hit `embed_failed → 2D`**, bitwise identical to the pre-refactor run — not a regression introduced here.
- The speed-up plateaus at 5× because 2D tensorization is Python/NumPy work that holds the GIL, so thread parallelism cannot fully overlap it.

RDKit's embed/MMFF release the GIL, which is why this uses a **thread pool** rather than a process pool; a process pool only adds serialization and memory-copy overhead.

## Re-running the benchmark

```shell
python benchmarks/bench_featurize.py --csv <csv> --per-bin 25 --workers 1,4,8
```

`--csv` needs a `SMILES` column and selects the first `--per-bin` rows per heavy-atom bin (`[5,15)`, `[15,25)`, `[25,40)`, `[40,60)`, `[60,80)`). The script only prints timings; it asserts no threshold. Without `--csv` it runs a small built-in smoke set.

## Determinism boundaries

- All randomness comes from a local `np.random.RandomState`; identical inputs and seeds are bitwise identical in serial and multi-threaded runs.
- Results are tied to the RDKit version (validated with `2025.09.1`).
- **With `timeout` enabled, results can differ across machines**: the same molecule may embed in 3D on a fast machine and degrade to 2D on a slow or loaded one. For strict reproducibility use `timeout=None`, or filter on `meta["status"]`.

## When to use the batch layer

- Large datasets with variable per-molecule cost (long-chain lipids and other hard-to-embed molecules) benefit from `featurize_batch(..., timeout=30.0)`, which keeps a bad molecule from stalling the whole run.
- When you need per-row auditing (which molecules degraded to 2D, which failed to parse), `meta` already answers it — no hand-written try/except needed.
- For a handful of molecules, or when the caller already has its own scheduler, call `smiles2unimolitem` directly.

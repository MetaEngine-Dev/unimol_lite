# Data containers (SMS-DataSpec adapter)

[中文](../zh-CN/11-dataspec.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 3 — Use

**Prerequisites:** [Installation](./01-installation.md), [Solution layers](./10-solution-layers.md), and the optional extra:

```bash
python -m pip install -e ".[dataspec]"      # polars + pyarrow
```

**Next:** [Solution layers](./10-solution-layers.md) · [CLI](./07-cli.md)

## What a container is

`SMS-DataSpec` is a container/contract for spectrum–molecule datasets: a
directory with three single-file parquet tables and a JSON envelope.

```text
<container>/
├── molecules.parquet      molecule rows (canonical SMILES + identity columns)
├── spectra.parquet        spectrum rows (peaks + acquisition context)
├── linker.parquet         spectrum <-> molecule associations (n:m) + provenance
└── container_meta.json    envelope: versions, provenance, counts, hash modes, fingerprint
```

- **Block = parquet row group.** Writing lays down a row group every `block_rows`
  rows (default 250 000); batched reading never lets a batch cross a row group.
- **Keys are content-bound.** A molecule key is its InChIKey (or the SHA-256 of
  the canonical SMILES as a fallback); a spectrum key is the hash of its peaks.
- **The envelope fingerprint is a mixing guard.** It hashes the contract fields
  (`spec_version`, `schema_version`, `hash_modes`, `metric`, `normalized`); two
  containers with different fingerprints must not be mixed.

This library implements the protocol itself (`unimol_lite.dataspec`); it does not
import another library's adapter. Compatibility is verified by tests:
round-trip, hash vectors, batched reads and conformance validation.

## Scope: the molecule side

| Aspect | This adapter |
| --- | --- |
| `molecules` | read **and** write (the focus of this library) |
| `spectra` | read; write only pass-through frames (keys supplied by the producer) |
| `linker` | read; write pass-through (`n:m` and provenance columns as given) |
| SPLASH spectrum hashing | **not implemented** — `splash` keys are opaque identifiers, conformance marks them unverified |
| `content_hash` spectrum hashing | implemented and verified on write |
| Pretraining LMDB | **out of scope** (upstream Uni-Mol format, exempt) |

## Record kinds

A molecule record is one molecule plus a payload kind, stored in the `unimol_*`
extension columns (the protocol allows extra columns next to the contract
columns). Two kinds are defined:

| Kind | Meaning | Payload |
| --- | --- | --- |
| `smiles` | featurize on read | none — `unimol_atoms` / `unimol_coords` / `unimol_coord_mode` are null |
| `pregenerated` | coordinates frozen at build time | `unimol_atoms`, `unimol_coords`, `unimol_coord_mode` |

Extension columns:

| Column | Type | Nullable | Meaning |
| --- | --- | --- | --- |
| `unimol_kind` | Utf8 | no (for rows written here) | `"smiles"` or `"pregenerated"` |
| `unimol_atoms` | List(Utf8) | yes | atom symbols **including hydrogens**, as produced by featurization |
| `unimol_coords` | List(Float64) | yes | flattened `x, y, z, ...`, length `3 * len(atoms)` |
| `unimol_coord_mode` | Utf8 | yes | provenance of the stored conformer: `"3d"` or `"2d"` |

Contract columns of `molecules.parquet` (fixed by the protocol, always written):

| Column | Type | Notes |
| --- | --- | --- |
| `molecule_key` | Utf8 | primary key, unique in the container |
| `key_mode` | Utf8 | `"inchikey"` or `"smiles_hash"` |
| `inchikey` | Utf8 | set, and equal to the key, when `key_mode="inchikey"` |
| `inchikey_skeleton` | Utf8 | first 25 characters of `inchikey` |
| `smiles` | Utf8 | canonical SMILES |
| `smiles_hash` | Utf8 | set, and equal to the key, when `key_mode="smiles_hash"` |
| `inchi` | Utf8 | full InChI (kept for adjudicating key ambiguities) |
| `name`, `formula` | Utf8 | optional metadata |
| `monoisotopic_mass` | Float64 | optional metadata |

Containers written by other tools have no `unimol_*` columns; their rows are read
as `smiles` records, because "canonical SMILES + identity" is exactly the
protocol's molecule contract.

## Reading

```python
from unimol_lite.dataspec import DataSpecContainer, iter_items, iter_records

container = DataSpecContainer("demo.msdata")     # validates the envelope on open
container.meta.counts                             # declared row counts
container.molecules()                             # whole table (small/fixture use)

for batch in container.iter_molecules_batches(65_536):
    ...                                           # batches respect row groups

for record in iter_records("demo.msdata", kind="pregenerated"):
    atoms, coords = record["unimol_atoms"], record["unimol_coords"]

for record, item, meta in iter_items("demo.msdata"):
    ...                                           # item is None when meta["status"] != "ok"
```

`record_to_item(record, ...)` converts one record to a model input item:
`pregenerated` records rebuild the item from the stored coordinates (no RDKit
embedding, reproducible across machines), `smiles` records go through the standard
featurization path.

## Writing

```python
from unimol_lite.dataspec import build_record, write_records

records = [
    build_record("CCO", kind="smiles"),
    build_record("c1ccccc1O", kind="pregenerated", coord_mode="auto"),
]
meta = write_records("demo.msdata", records, shuffle_strategy="row_group_shuffle")
```

`build_record` canonicalises the SMILES with RDKit, keys the molecule by InChIKey
(falling back to the SMILES hash), fills formula/mass/InChI when available, and —
for `pregenerated` — generates one conformer and stores it. `write_records`
derives the container-level `hash_modes` from the rows, refuses duplicate keys,
writes the three tables chunked by `block_rows`, and finalises the envelope
(counts, timestamp, fingerprint).

`shuffle_strategy` is a declaration the protocol requires for training use; this
adapter writes files in insertion order, so `"row_group_shuffle"` (shuffle whole
blocks when reading) or `"arrow_mmap_random_row"` describe what the consumer does.
`metric` / `normalized` are reserved placeholders that participate in the
fingerprint — do not set them unless the container really is a vector store.

## Conformance validation

```python
from unimol_lite.dataspec import validate_container

report = validate_container("demo.msdata")
print(report.summary())
if not report.ok:
    for issue in report.errors():
        print(issue.code, issue.table, issue.message)
```

Errors refuse the container (missing table or column, wrong dtype, duplicate or
null keys, inconsistent key mode, hash mismatch, dangling linker reference,
envelope fingerprint mismatch, counts mismatch, illegal extension payload).
Warnings are runnable-but-suspicious: `molecules.inchi_missing` (no full InChI to
adjudicate an ambiguity) and `spectra.key_unverified` (SPLASH keys this library
cannot recompute).

## Hashing

| Function | Rule |
| --- | --- |
| `smiles_hash(smiles)` | `sha256(canonical SMILES bytes)` — 64 hex chars |
| `inchikey_skeleton(inchikey)` | first 25 characters (connectivity + stereo block) |
| `spectrum_content_hash(peaks, *, precursor_mz, precursor_type, ionization_mode, collision_energy)` | peaks sorted by m/z ascending then intensity descending, rendered `f"{mz:.4f}:{intensity:.2f}"`, joined by `\|`, plus the four context fields (4/2 decimals, missing = empty), SHA-256 |

SPLASH is intentionally absent: this library never produces spectra, so it never
needs to derive a spectrum key. A container that carries SPLASH keys is still
readable — the keys are treated as opaque identifiers.

## Interoperability notes

- **Arrow type variants.** Logical `Utf8` / `List(Float64)` columns may appear as
  `large_string` / `large_list` (what polars writes) or as `string` / `list`.
  Readers accept both; type variants never justify refusing a container.
- **Zero-row tables** are legal and must exist as schema-only parquet files; their
  hash-mode declaration is not reconciled against rows.
- **Extras columns** are preserved on read and never rejected.

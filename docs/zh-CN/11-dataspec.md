# 数据容器（SMS-DataSpec 适配器）

[English](../en/11-dataspec.md) | **简体中文** · [文档索引](./index.md)

**级别：** 3 — 使用

**前置：** [安装](./01-installation.md)、[解决方案四层结构](./10-solution-layers.md)，以及可选依赖：

```bash
python -m pip install -e ".[dataspec]"      # polars + pyarrow
```

**下一页：** [四层结构](./10-solution-layers.md) · [命令行接口](./07-cli.md)

## 容器是什么

`SMS-DataSpec` 是谱图–分子数据集的容器与契约：一个目录，内含三张单文件 parquet 表与一个 JSON envelope。

```text
<container>/
├── molecules.parquet      分子行（canonical SMILES + 身份列）
├── spectra.parquet        谱图行（峰 + 采集上下文）
├── linker.parquet         谱图 ↔ 分子关联（n:m）+ 溯源
└── container_meta.json    envelope：版本、来源、行数、哈希模式、契约指纹
```

- **块 = parquet row group**：写入按 `block_rows` 行切一个 row group（默认 250 000）；批式读取的批不会跨越 row group。
- **键与内容绑定**：分子键是 InChIKey（不可得时回退到 canonical SMILES 的 SHA-256）；谱图键是其峰内容的哈希。
- **envelope 指纹是混用闸门**：对契约字段（`spec_version`、`schema_version`、`hash_modes`、`metric`、`normalized`）做哈希；指纹不同的两个容器不得混用。

本库自己实现该协议（`unimol_lite.dataspec`），不 import 其他库的适配器。兼容性由测试保证：roundtrip、哈希向量、批式读取、一致性校验。

## 范围：分子侧

| 面 | 本适配器 |
| --- | --- |
| `molecules` | 读**和**写（本库的着力点） |
| `spectra` | 读；写只做透传（键由生产者提供） |
| `linker` | 读；写透传（`n:m` 与溯源列原样） |
| SPLASH 谱图哈希 | **不实现**——`splash` 键按不透明标识透传，一致性校验标为未验证 |
| `content_hash` 谱图哈希 | 实现，写入时校验 |
| 预训练 LMDB | **不在范围内**（上游 Uni-Mol 格式，豁免） |

## 记录类型（kinds）

一条分子记录 = 一个分子 + 一种负载类型，负载存放于 `unimol_*` 扩展列（协议允许扩展列与契约列并存）。共定义两种：

| 类型 | 含义 | 负载 |
| --- | --- | --- |
| `smiles` | 读取时特征化 | 无——`unimol_atoms` / `unimol_coords` / `unimol_coord_mode` 均为 null |
| `pregenerated` | 构建期冻结构象 | `unimol_atoms`、`unimol_coords`、`unimol_coord_mode` |

扩展列：

| 列 | 类型 | 可空 | 含义 |
| --- | --- | --- | --- |
| `unimol_kind` | Utf8 | 否（本库写入的行） | `"smiles"` 或 `"pregenerated"` |
| `unimol_atoms` | List(Utf8) | 是 | 原子符号，**含氢**，与特征化输出一致 |
| `unimol_coords` | List(Float64) | 是 | 展平的 `x, y, z, ...`，长度 `3 * len(atoms)` |
| `unimol_coord_mode` | Utf8 | 是 | 存储构象的来源：`"3d"` 或 `"2d"` |

`molecules.parquet` 的契约列（协议固定，本库总是写全）：

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `molecule_key` | Utf8 | 主键，容器内唯一 |
| `key_mode` | Utf8 | `"inchikey"` 或 `"smiles_hash"` |
| `inchikey` | Utf8 | `key_mode="inchikey"` 时非空且等于键 |
| `inchikey_skeleton` | Utf8 | `inchikey` 前 25 字符 |
| `smiles` | Utf8 | canonical SMILES |
| `smiles_hash` | Utf8 | `key_mode="smiles_hash"` 时非空且等于键 |
| `inchi` | Utf8 | InChI 全文（键歧义时的裁决依据） |
| `name`、`formula` | Utf8 | 可选元数据 |
| `monoisotopic_mass` | Float64 | 可选元数据 |

其他工具写出的容器没有 `unimol_*` 列；其行按 `smiles` 记录读取——「canonical SMILES + 身份」正是协议的分子契约。

## 读取

```python
from unimol_lite.dataspec import DataSpecContainer, iter_items, iter_records

container = DataSpecContainer("demo.msdata")     # 打开即校验 envelope
container.meta.counts                             # envelope 声明的行数
container.molecules()                             # 整表读取（小表/夹具场景）

for batch in container.iter_molecules_batches(65_536):
    ...                                           # 批不跨 row group

for record in iter_records("demo.msdata", kind="pregenerated"):
    atoms, coords = record["unimol_atoms"], record["unimol_coords"]

for record, item, meta in iter_items("demo.msdata"):
    ...                                           # meta["status"] != "ok" 时 item 为 None
```

`record_to_item(record, ...)` 把单条记录转成模型输入：`pregenerated` 记录直接用存储坐标重建（不做 RDKit 嵌入，跨机器可复现）；`smiles` 记录走标准特征化路径。

## 写入

```python
from unimol_lite.dataspec import build_record, write_records

records = [
    build_record("CCO", kind="smiles"),
    build_record("c1ccccc1O", kind="pregenerated", coord_mode="auto"),
]
meta = write_records("demo.msdata", records, shuffle_strategy="row_group_shuffle")
```

`build_record` 用 RDKit 规范化 SMILES，以 InChIKey 作键（不可得时回退 SMILES 哈希），尽量填 formula/质量/InChI；`pregenerated` 还会生成一个构象并存入。`write_records` 从行推导容器级 `hash_modes`、拒绝重复键、按 `block_rows` 分块写三表，并定稿 envelope（行数、时间戳、指纹）。

`shuffle_strategy` 是协议要求训练用途声明的字段；本适配器按插入顺序落盘，因此 `"row_group_shuffle"`（读取时按块洗牌）或 `"arrow_mmap_random_row"` 描述的是消费侧行为。`metric` / `normalized` 是参与指纹的预留占位——除非容器真是向量库，否则不要设置。

## 一致性校验

```python
from unimol_lite.dataspec import validate_container

report = validate_container("demo.msdata")
print(report.summary())
if not report.ok:
    for issue in report.errors():
        print(issue.code, issue.table, issue.message)
```

错误级（拒绝容器）：表或列缺失、dtype 不符、键重复或为空、key_mode 自相矛盾、哈希不匹配、linker 悬挂引用、envelope 指纹不自洽、counts 不符、扩展负载非法。
警告级（可运行但可疑）：`molecules.inchi_missing`（缺 InChI 全文，歧义无从裁决）、`spectra.key_unverified`（本库无法重算的 SPLASH 键）。

## 哈希

| 函数 | 规则 |
| --- | --- |
| `smiles_hash(smiles)` | `sha256(canonical SMILES 字节)` —— 64 位十六进制 |
| `inchikey_skeleton(inchikey)` | 前 25 字符（连接性段 + 立体/同位素段） |
| `spectrum_content_hash(peaks, *, precursor_mz, precursor_type, ionization_mode, collision_energy)` | 峰按 m/z 升序、强度降序排序，逐峰 `f"{mz:.4f}:{intensity:.2f}"` 以 `\|` 连接，再拼四个上下文字段（4/2 位小数，缺失记空），取 SHA-256 |

SPLASH 刻意不实现：本库从不产出谱图，也就不需要推导谱图键。带 SPLASH 键的容器依然可读——键按不透明标识处理。

## 互操作说明

- **Arrow 类型变体**：逻辑 `Utf8` / `List(Float64)` 列可能以 `large_string` / `large_list`（polars 写出）或 `string` / `list` 出现。读取方两者都接受；类型变体不构成拒绝容器的理由。
- **零行表**是合法的，且必须存在（仅含 schema 的 parquet）；其哈希模式声明不与行级对账。
- **扩展列**读取时保留，不得因扩展列报错。

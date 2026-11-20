"""C8 — conformance validation: a legal container passes, each break is reported.

Every case breaks exactly one thing (usually by rewriting a table or patching the
envelope after the fact) and asserts that the corresponding issue code appears.
"""

from __future__ import annotations

import polars as pl
import pytest

# This suite needs the optional dataspec backend; skip cleanly without it.
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from helpers import (
    molecules_frame,
    overwrite_envelope,
    patch_envelope,
    rewrite_table,
)
from unimol_lite.dataspec import (
    ConformanceReport,
    validate_container,
)
from unimol_lite.dataspec.schema import (
    LINKER_SCHEMA,
    TABLE_LINKER,
    TABLE_MOLECULES,
    TABLE_SPECTRA,
)


class TestLegalContainers:
    def test_fixture_container_passes(self, make_container):
        report = validate_container(make_container().root)
        assert isinstance(report, ConformanceReport)
        assert report.ok, report.summary()
        assert report.errors() == []

    def test_report_summary_is_readable(self, make_container):
        summary = validate_container(make_container(name="summary").root).summary()
        assert summary.startswith("PASS")
        assert "error(s)" in summary

    def test_mixed_modes_container_passes(self, make_container):
        assert validate_container(make_container(name="mixed-modes", mixed=True).root).ok


class TestTableBreaks:
    def test_missing_table_file(self, make_container):
        bundle = make_container(name="no-spectra")
        (bundle.root / f"{TABLE_SPECTRA}.parquet").unlink()
        report = validate_container(bundle.root)
        assert "table.missing" in report.codes()
        assert not report.ok

    def test_missing_contract_column(self, make_container):
        bundle = make_container(name="no-column")
        rewrite_table(bundle.root, TABLE_MOLECULES, bundle.molecules.drop("formula"))
        assert "table.column_missing" in validate_container(bundle.root).codes()

    def test_wrong_contract_dtype(self, make_container):
        bundle = make_container(name="bad-dtype")
        tampered = bundle.molecules.with_columns(
            pl.Series("monoisotopic_mass", ["94.04", "46.04"])
        )
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "table.column_type" in validate_container(bundle.root).codes()

    def test_missing_key_column_skips_row_checks(self, make_container):
        bundle = make_container(name="no-key")
        rewrite_table(bundle.root, TABLE_MOLECULES, bundle.molecules.drop("molecule_key"))
        codes = validate_container(bundle.root).codes()
        assert "table.column_missing" in codes
        assert "molecules.key_null" not in codes  # row-level checks are skipped, not crashed


class TestMoleculeRowBreaks:
    def test_duplicate_molecule_key(self, make_container):
        bundle = make_container(name="dup")
        duplicated = pl.concat([bundle.molecules, bundle.molecules.head(1)])
        rewrite_table(bundle.root, TABLE_MOLECULES, duplicated)
        assert "molecules.key_dup" in validate_container(bundle.root).codes()

    def test_null_molecule_key(self, make_container):
        bundle = make_container(name="null-key")
        tampered = bundle.molecules.with_columns(
            pl.Series("molecule_key", [None, bundle.molecules["molecule_key"][1]])
        )
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "molecules.key_null" in validate_container(bundle.root).codes()

    def test_illegal_key_mode(self, make_container):
        bundle = make_container(name="bad-mode")
        tampered = bundle.molecules.with_columns(pl.Series("key_mode", ["md5", "inchikey"]))
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "molecules.key_mode_invalid" in validate_container(bundle.root).codes()

    def test_inchikey_mode_with_mismatched_key(self, make_container):
        bundle = make_container(name="key-mismatch")
        tampered = bundle.molecules.with_columns(
            pl.Series("molecule_key", [bundle.molecules["molecule_key"][0], "0" * 27])
        )
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "molecules.key_mode_inconsistent" in validate_container(bundle.root).codes()

    def test_inchikey_mode_with_smiles_hash_set(self, make_container):
        bundle = make_container(name="hash-set")
        tampered = bundle.molecules.with_columns(pl.Series("smiles_hash", ["a" * 64, None]))
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "molecules.key_mode_inconsistent" in validate_container(bundle.root).codes()

    def test_wrong_smiles_hash(self, make_container):
        bundle = make_container(name="wrong-hash", mixed=True)
        tampered = bundle.molecules.with_columns(
            pl.Series("molecule_key", [bundle.molecules["molecule_key"][0], "b" * 64]),
            pl.Series("smiles_hash", [None, "b" * 64]),
        )
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "molecules.smiles_hash_mismatch" in validate_container(bundle.root).codes()

    def test_wrong_skeleton(self, make_container):
        bundle = make_container(name="wrong-skeleton")
        tampered = bundle.molecules.with_columns(
            pl.Series(
                "inchikey_skeleton",
                ["AAAAAUHFFAUHFFAUHFFAUHFFA"] + bundle.molecules["inchikey_skeleton"].to_list()[1:],
            )
        )
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "molecules.skeleton_mismatch" in validate_container(bundle.root).codes()

    def test_empty_smiles(self, make_container):
        bundle = make_container(name="empty-smiles")
        tampered = bundle.molecules.with_columns(pl.Series("smiles", ["", "CCO"]))
        rewrite_table(bundle.root, TABLE_MOLECULES, tampered)
        assert "molecules.smiles_null" in validate_container(bundle.root).codes()

    def test_missing_inchi_is_only_a_warning(self, make_container):
        bundle = make_container(name="no-inchi", molecules=molecules_frame(with_inchi=False))
        report = validate_container(bundle.root)
        assert report.ok
        assert "molecules.inchi_missing" in report.codes()
        assert any(issue.severity == "warning" for issue in report.warnings())


class TestSpectraRowBreaks:
    def test_wrong_content_hash_key(self, make_container):
        bundle = make_container(name="wrong-spectrum-key")
        tampered = bundle.spectra.with_columns(pl.Series("spectrum_key", ["c" * 64] * 3))
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        assert "spectra.key_mismatch" in validate_container(bundle.root).codes()

    def test_duplicate_spectrum_key(self, make_container):
        bundle = make_container(name="dup-spectrum")
        tampered = bundle.spectra.with_columns(
            pl.Series("spectrum_key", [bundle.spectra["spectrum_key"][0]] * 3)
        )
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        assert "spectra.key_dup" in validate_container(bundle.root).codes()

    def test_empty_peak_list(self, make_container):
        bundle = make_container(name="empty-peaks")
        tampered = bundle.spectra.with_columns(pl.Series("peaks_mz", [[], [1.0], [2.0]]))
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        assert "spectra.peaks_invalid" in validate_container(bundle.root).codes()

    def test_non_positive_max_intensity(self, make_container):
        bundle = make_container(name="zero-intensity")
        tampered = bundle.spectra.with_columns(
            pl.Series("peaks_intensity", [[0.0, 0.0, 0.0], [1.0], [1.0]])
        )
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        assert "spectra.peaks_invalid" in validate_container(bundle.root).codes()

    def test_illegal_hash_mode(self, make_container):
        bundle = make_container(name="bad-hash-mode")
        tampered = bundle.spectra.with_columns(pl.Series("hash_mode", ["md5", "content_hash", "content_hash"]))
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        assert "spectra.hash_mode_invalid" in validate_container(bundle.root).codes()

    def test_nan_peak_is_reported_for_splash_keys(self, make_container):
        """P3-a：NaN/Inf 峰在 splash 分支同样报 peaks_invalid（不再透传逃逸）。"""
        bundle = make_container(name="nan-peaks")
        tampered = bundle.spectra.with_columns(
            pl.Series("peaks_intensity", [[float("nan"), 1.0, 2.0], [1.0], [1.0]])
        )
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        report = validate_container(bundle.root)
        assert "spectra.peaks_invalid" in report.codes()

    def test_inf_mz_is_reported_for_content_hash_keys(self, make_container):
        """P3-a：Inf m/z 在 content_hash 分支报 peaks_invalid（哈希重算前的有限性门）。"""
        bundle = make_container(name="inf-mz")
        tampered = bundle.spectra.with_columns(
            pl.Series("peaks_mz", [[float("inf"), 2.0], [1.0], [1.0]])
        )
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        report = validate_container(bundle.root)
        assert "spectra.peaks_invalid" in report.codes()

    def test_null_intensity_normalized(self, make_container):
        """P3-d：intensity_normalized 空值 → spectra.intensity_normalized_null。"""
        bundle = make_container(name="null-intensity-flag")
        rows = bundle.spectra.height
        tampered = bundle.spectra.with_columns(
            pl.Series("intensity_normalized", [None] * rows, dtype=pl.Boolean)
        )
        rewrite_table(bundle.root, TABLE_SPECTRA, tampered)
        assert "spectra.intensity_normalized_null" in validate_container(bundle.root).codes()

    def test_illegal_shuffle_strategy(self, make_container, make_meta):
        """P3-d：envelope.shuffle_strategy 非法值 → 拒绝容器（fail-closed）。

        分层防御实测：unimol_lite 的 envelope 模型对 shuffle_strategy 做枚举校验，
        非法值在读侧 ``envelope.invalid``（pydantic 校验）即被拒——conformance 的
        ``envelope.shuffle_strategy_invalid`` 专用检查为冗余内门（仅当校验器被
        直接喂跳过 envelope 校验的 meta 时可达）。本用例钉住拒绝行为。
        """
        bundle = make_container(name="bad-shuffle", meta=make_meta(shuffle_strategy="bogus"))
        assert "envelope.invalid" in validate_container(bundle.root).codes()
        assert not validate_container(bundle.root).ok


class TestUnimolExtensionBreaks:
    def _with_extension(self, molecules: pl.DataFrame, kind: str, atoms, coords) -> pl.DataFrame:
        rows = molecules.height
        return molecules.with_columns(
            pl.Series("unimol_kind", [kind] + ["smiles"] * (rows - 1)),
            pl.Series("unimol_atoms", [atoms] + [None] * (rows - 1), dtype=pl.List(pl.String)),
            pl.Series("unimol_coords", [coords] + [None] * (rows - 1), dtype=pl.List(pl.Float64)),
            pl.Series("unimol_coord_mode", ["3d"] + [None] * (rows - 1)),
        )

    def test_valid_pregenerated_payload_passes(self, make_container):
        molecules = self._with_extension(molecules_frame(), "pregenerated", ["C", "C", "O"], [0.0] * 9)
        bundle = make_container(name="ext-ok", molecules=molecules)
        assert validate_container(bundle.root).ok

    def test_coordinate_length_mismatch(self, make_container):
        molecules = self._with_extension(molecules_frame(), "pregenerated", ["C", "C", "O"], [0.0] * 8)
        bundle = make_container(name="ext-bad-coords", molecules=molecules)
        assert "extension.unimol_payload_invalid" in validate_container(bundle.root).codes()

    def test_unknown_kind(self, make_container):
        molecules = self._with_extension(molecules_frame(), "tensors", ["C"], [0.0] * 3)
        bundle = make_container(name="ext-bad-kind", molecules=molecules)
        assert "extension.unimol_kind_invalid" in validate_container(bundle.root).codes()


class TestEnvelopeBreaks:
    def test_counts_mismatch(self, make_container):
        bundle = make_container(name="counts")
        patch_envelope(bundle.root, counts={"molecules": 5, "spectra": 3, "linker": 4})
        report = validate_container(bundle.root)
        assert "envelope.counts_mismatch" in report.codes()
        assert not report.ok

    def test_fingerprint_mismatch(self, make_container):
        bundle = make_container(name="fingerprint")
        patch_envelope(bundle.root, recompute_fingerprint=False, normalized=True)
        assert "envelope.fingerprint_mismatch" in validate_container(bundle.root).codes()

    def test_missing_envelope_field(self, make_container):
        bundle = make_container(name="no-builder")
        overwrite_envelope(bundle.root, builder=None)
        report = validate_container(bundle.root)
        assert report.codes() == ["envelope.missing_field"]
        assert not report.ok

    def test_unknown_major_version(self, make_container):
        bundle = make_container(name="future-version")
        patch_envelope(bundle.root, spec_version="2.0.0")
        codes = validate_container(bundle.root).codes()
        assert "envelope.unknown_spec_version" in codes

    def test_hash_modes_contradicting_rows(self, make_container):
        bundle = make_container(name="mode-contradiction")
        patch_envelope(bundle.root, hash_modes={"spectra": "splash", "molecules": "inchikey"})
        assert "envelope.hash_modes_mismatch" in validate_container(bundle.root).codes()

    def test_empty_linker_table_is_fine(self, make_container):
        bundle = make_container(name="no-linker", linker=pl.DataFrame(schema=LINKER_SCHEMA))
        assert validate_container(bundle.root).ok


@pytest.mark.parametrize("table", [TABLE_MOLECULES, TABLE_SPECTRA, TABLE_LINKER])
def test_each_table_file_missing_is_reported_once(make_container, table):
    bundle = make_container(name=f"missing-{table}")
    (bundle.root / f"{table}.parquet").unlink()
    report = validate_container(bundle.root)
    missing = [issue for issue in report.errors() if issue.code == "table.missing"]
    assert len(missing) == 1
    assert missing[0].table == table

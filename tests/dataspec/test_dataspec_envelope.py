"""C5 / C7 — envelope completeness, reconciliation and the mixing guard.

C5: required fields present, counts reconciled against the tables, missing
fields or wrong counts refused.
C7: containers with different contract fields (``metric`` / ``normalized`` /
``schema_version``) must be refused when mixed, via the fingerprint.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

# This suite needs the optional dataspec backend; skip cleanly without it.
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from helpers import (
    molecules_frame,
    overwrite_envelope,
    patch_envelope,
    read_envelope_json,
    rewrite_table,
)

from unimol_lite.dataspec import (
    CONTAINER_META_FILENAME,
    SCHEMA_VERSION,
    SPEC_VERSION,
    BuilderInfo,
    ContainerMeta,
    DataSpecMismatchError,
    DataSpecValidationError,
    HashModes,
    SourceInfo,
    assert_compatible,
    container_fingerprint,
    read_container_meta,
    validate_container,
)
from unimol_lite.dataspec.schema import LINKER_SCHEMA, SPECTRA_SCHEMA


class TestEnvelopeRead:
    def test_all_required_fields_are_present(self, make_container):
        bundle = make_container()
        payload = read_envelope_json(bundle.root)
        for key in (
            "spec_version",
            "schema_version",
            "source",
            "builder",
            "counts",
            "created_at",
            "hash_modes",
            "fingerprint",
        ):
            assert payload[key] is not None, key
        assert payload["spec_version"] == SPEC_VERSION
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["builder"]["name"] == "dataspec-tests"

    def test_counts_reconcile_with_rows(self, make_container):
        bundle = make_container()
        counts = read_container_meta(bundle.root).counts
        assert (counts.molecules, counts.spectra, counts.linker) == (2, 3, 4)

    def test_shuffle_strategy_declaration_is_readable(self, make_container, make_meta):
        meta = make_meta(shuffle_strategy="row_group_shuffle")
        bundle = make_container(name="shuffle", meta=meta)
        assert read_container_meta(bundle.root).shuffle_strategy == "row_group_shuffle"

    def test_zero_row_container_passes_with_any_legal_declaration(self, make_container):
        from unimol_lite.dataspec import empty_table

        bundle = make_container(
            name="empty",
            molecules=empty_table("molecules"),
            spectra=empty_table("spectra"),
            linker=empty_table("linker"),
        )
        assert validate_container(bundle.root).ok
        assert bundle.meta.counts.molecules == 0


class TestEnvelopeFailures:
    @pytest.mark.parametrize("field", ["spec_version", "schema_version", "source", "builder", "counts", "hash_modes"])
    def test_missing_required_field_is_refused(self, make_container, field):
        bundle = make_container(name=f"missing-{field}")
        overwrite_envelope(bundle.root, **{field: None})
        with pytest.raises(DataSpecValidationError):
            read_container_meta(bundle.root)
        assert "envelope.missing_field" in validate_container(bundle.root).codes()

    def test_missing_envelope_file_is_refused(self, make_container):
        bundle = make_container(name="no-envelope")
        (bundle.root / CONTAINER_META_FILENAME).unlink()
        with pytest.raises(DataSpecValidationError):
            read_container_meta(bundle.root)
        assert "envelope.missing" in validate_container(bundle.root).codes()

    def test_unparsable_envelope_is_refused(self, make_container):
        bundle = make_container(name="bad-json")
        (bundle.root / CONTAINER_META_FILENAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(DataSpecValidationError):
            read_container_meta(bundle.root)
        assert "envelope.unparseable" in validate_container(bundle.root).codes()

    def test_unknown_major_version_is_refused(self, make_container):
        bundle = make_container(name="future")
        patch_envelope(bundle.root, recompute_fingerprint=True, spec_version="2.0.0")
        with pytest.raises(DataSpecValidationError):
            read_container_meta(bundle.root)
        assert "envelope.unknown_spec_version" in validate_container(bundle.root).codes()

    def test_illegal_hash_modes_are_refused(self, make_container):
        bundle = make_container(name="bad-modes")
        overwrite_envelope(bundle.root, hash_modes={"spectra": "md5", "molecules": "inchikey"})
        with pytest.raises(DataSpecValidationError):
            read_container_meta(bundle.root)
        assert "envelope.invalid" in validate_container(bundle.root).codes()

    def test_illegal_shuffle_strategy_is_refused(self, make_container):
        bundle = make_container(name="bad-shuffle")
        overwrite_envelope(bundle.root, shuffle_strategy="random")
        with pytest.raises(DataSpecValidationError):
            read_container_meta(bundle.root)

    def test_wrong_counts_are_reported(self, make_container):
        bundle = make_container(name="bad-counts")
        patch_envelope(bundle.root, counts={"molecules": 99, "spectra": 3, "linker": 4})
        report = validate_container(bundle.root)
        assert not report.ok
        assert "envelope.counts_mismatch" in report.codes()

    def test_tampered_fingerprint_is_refused(self, make_container):
        bundle = make_container(name="tampered")
        patch_envelope(bundle.root, recompute_fingerprint=False, metric="cosine")
        with pytest.raises(DataSpecValidationError):
            read_container_meta(bundle.root)
        assert "envelope.fingerprint_mismatch" in validate_container(bundle.root).codes()

    def test_writer_refuses_contradicted_hash_modes(self, tmp_path, make_meta):
        from unimol_lite.dataspec import DataSpecError, write_container

        meta = make_meta(hash_modes=HashModes(spectra="content_hash", molecules="smiles_hash"))
        with pytest.raises(DataSpecError):
            write_container(
                tmp_path / "contradiction",
                molecules_frame(),
                pl.DataFrame(schema=SPECTRA_SCHEMA),
                pl.DataFrame(schema=LINKER_SCHEMA),
                meta=meta,
            )


class TestFingerprint:
    def test_fingerprint_is_sha256_of_canonical_contract_json(self):
        meta = ContainerMeta(
            spec_version=SPEC_VERSION,
            schema_version=SCHEMA_VERSION,
            source=SourceInfo(),
            builder=BuilderInfo(name="x"),
            hash_modes=HashModes(spectra="content_hash", molecules="inchikey"),
        )
        canonical = json.dumps(
            {
                "spec_version": SPEC_VERSION,
                "schema_version": SCHEMA_VERSION,
                "hash_modes": {"spectra": "content_hash", "molecules": "inchikey"},
                "metric": None,
                "normalized": None,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        import hashlib

        expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert container_fingerprint(meta) == expected

    def test_fingerprint_ignores_provenance_and_counts(self, make_container, make_meta):
        base = make_meta()
        other = make_meta(
            source=SourceInfo(digest="sha256:" + "1" * 64),
            builder=BuilderInfo(name="another-builder", version="9.9"),
            shuffle_strategy="pre_shuffled_blocks",
        )
        assert container_fingerprint(base) == container_fingerprint(other)

    def test_stored_fingerprint_matches_recomputation(self, make_container):
        bundle = make_container(name="fp")
        meta = read_container_meta(bundle.root)
        assert meta.fingerprint == container_fingerprint(meta)


class TestMixingGuard:
    def _meta(self, **overrides) -> ContainerMeta:
        fields: dict[str, object] = {
            "spec_version": SPEC_VERSION,
            "schema_version": SCHEMA_VERSION,
            "source": SourceInfo(),
            "builder": BuilderInfo(name="x"),
            "hash_modes": HashModes(spectra="content_hash", molecules="inchikey"),
        }
        fields.update(overrides)
        return ContainerMeta(**fields)

    def test_identical_contracts_are_compatible(self):
        assert_compatible(self._meta(), self._meta())
        assert_compatible(self._meta(), container_fingerprint(self._meta()))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"metric": "cosine"},
            {"normalized": True},
            {"schema_version": "2"},
            {"hash_modes": HashModes(spectra="splash", molecules="inchikey")},
            {"hash_modes": HashModes(spectra="content_hash", molecules="smiles_hash")},
        ],
    )
    def test_contract_differences_are_refused(self, overrides):
        with pytest.raises(DataSpecMismatchError):
            assert_compatible(self._meta(), self._meta(**overrides))

    def test_container_api_exposes_the_guard(self, make_container, make_meta):
        bundle = make_container(name="guard")
        bundle.container.assert_compatible_with(read_container_meta(bundle.root))
        with pytest.raises(DataSpecMismatchError):
            bundle.container.assert_compatible_with(container_fingerprint(make_meta(metric="dot")))

    def test_read_refuses_container_whose_tables_disagree_with_envelope(self, make_container):
        """Rows rewritten after the fact are only visible to the validator."""
        bundle = make_container(name="row-drift")
        bigger = pl.concat([bundle.molecules, molecules_frame().head(1)])
        rewrite_table(bundle.root, "molecules", bigger)
        assert "envelope.counts_mismatch" in validate_container(bundle.root).codes()

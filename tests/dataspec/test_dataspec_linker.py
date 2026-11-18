"""C4 — linker integrity: n:m is legal, dangling references are not.

The linker is an observation table: one spectrum may belong to several molecules
and one molecule may own several spectra; the same ``(spectrum_key,
molecule_key)`` pair may even repeat (one row per observation).  What the
protocol forbids is a reference that hits nothing.
"""

from __future__ import annotations

import polars as pl
import pytest

# This suite needs the optional dataspec backend; skip cleanly without it.
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from helpers import (
    linker_frame,
    molecules_frame,
    read_envelope_json,
    rewrite_table,
    spectra_frame,
)

from unimol_lite.dataspec import validate_container
from unimol_lite.dataspec.conformance import validate_container as validate


def _write_with_linker(make_container, name: str, linker: pl.DataFrame, **kwargs):
    return make_container(name=name, linker=linker, **kwargs)


class TestLinkerNm:
    def test_many_to_many_is_legal(self, make_container):
        spectra = spectra_frame()
        molecules = molecules_frame()
        linker = linker_frame(spectra, molecules)
        bundle = make_container(name="nm", spectra=spectra, molecules=molecules, linker=linker)
        assert validate_container(bundle.root).ok
        # one spectrum, two molecules
        first = spectra["spectrum_key"][0]
        owners = bundle.linker.filter(pl.col("spectrum_key") == first)["molecule_key"].to_list()
        assert len(owners) == 2
        # one molecule, several spectra
        ethanol = molecules["molecule_key"][1]
        assert bundle.linker.filter(pl.col("molecule_key") == ethanol).height == 2

    def test_repeated_pair_is_allowed(self, make_container):
        spectra = spectra_frame()
        molecules = molecules_frame()
        linker = linker_frame(spectra, molecules)
        duplicated = pl.concat([linker, linker.head(1)])
        bundle = make_container(name="repeat", spectra=spectra, molecules=molecules, linker=duplicated)
        assert validate_container(bundle.root).ok

    def test_source_provenance_columns_are_optional(self, make_container):
        spectra = spectra_frame()
        molecules = molecules_frame()
        linker = pl.DataFrame({
            "spectrum_key": [spectra["spectrum_key"][0]],
            "molecule_key": [molecules["molecule_key"][0]],
            "source_library": [None],
            "source_file": [None],
            "acquisition_context": [None],
        })
        bundle = make_container(name="bare-linker", spectra=spectra, molecules=molecules, linker=linker)
        assert validate_container(bundle.root).ok


class TestLinkerReferentialIntegrity:
    def test_dangling_spectrum_key_is_an_error(self, make_container):
        spectra = spectra_frame()
        molecules = molecules_frame()
        linker = pl.DataFrame({
            "spectrum_key": ["deadbeef" * 8],
            "molecule_key": [molecules["molecule_key"][0]],
            "source_library": [None],
            "source_file": [None],
            "acquisition_context": [None],
        })
        bundle = make_container(name="dangling-spectrum", spectra=spectra, molecules=molecules, linker=linker)
        report = validate(bundle.root)
        assert not report.ok
        assert "linker.fk_spectra_missing" in report.codes()

    def test_dangling_molecule_key_is_an_error(self, make_container):
        spectra = spectra_frame()
        molecules = molecules_frame()
        linker = pl.DataFrame({
            "spectrum_key": [spectra["spectrum_key"][0]],
            "molecule_key": ["0" * 64],
            "source_library": [None],
            "source_file": [None],
            "acquisition_context": [None],
        })
        bundle = make_container(name="dangling-molecule", spectra=spectra, molecules=molecules, linker=linker)
        report = validate(bundle.root)
        assert not report.ok
        assert "linker.fk_molecules_missing" in report.codes()

    def test_null_keys_reported(self, make_container):
        spectra = spectra_frame()
        molecules = molecules_frame()
        linker = pl.DataFrame(
            {
                "spectrum_key": [spectra["spectrum_key"][0], None],
                "molecule_key": [molecules["molecule_key"][0], molecules["molecule_key"][1]],
                "source_library": [None, None],
                "source_file": [None, None],
                "acquisition_context": [None, None],
            },
            schema={
                "spectrum_key": pl.String,
                "molecule_key": pl.String,
                "source_library": pl.String,
                "source_file": pl.String,
                "acquisition_context": pl.String,
            },
        )
        bundle = make_container(name="null-keys", spectra=spectra, molecules=molecules, linker=linker)
        assert "linker.key_null" in validate(bundle.root).codes()

    def test_mutated_linker_is_caught_after_the_fact(self, make_container):
        """A container can be edited on disk; the validator must notice."""
        bundle = make_container(name="mutated")
        tampered = bundle.linker.with_columns(pl.lit("not-a-molecule-key").alias("molecule_key"))
        rewrite_table(bundle.root, "linker", tampered)
        # counts still agree (same number of rows); the FK check catches the rest
        assert read_envelope_json(bundle.root)["counts"]["linker"] == tampered.height
        assert "linker.fk_molecules_missing" in validate(bundle.root).codes()

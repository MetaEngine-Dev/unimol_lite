"""C2 / C3 — hashing on the molecule side, plus the content-hash fallback path.

Expected values are external literals: ``sha256sum`` of the exact byte strings the
protocol prescribes, so the tests lock the algorithm rather than re-deriving it
from the implementation under test.

C2 (SPLASH determinism) is **out of scope by design**: this library is
molecule-domain and deliberately does not implement SPLASH.  What is tested
instead is the property that matters here — spectrum keys of ``splash`` rows are
carried through untouched and reported as unverified, and ``content_hash`` rows
are deterministic.
"""

from __future__ import annotations

import pytest

# This suite needs the optional dataspec backend; skip cleanly without it.
pytest.importorskip("polars")
pytest.importorskip("pyarrow")

from helpers import molecules_frame, spectra_frame

from unimol_lite.dataspec import (
    inchikey_skeleton,
    smiles_hash,
    spectrum_content_hash,
    validate_peaks,
)
from unimol_lite.dataspec.conformance import validate_container
from unimol_lite.dataspec.schema import TABLE_SPECTRA

#: sha256("CCO") — computed outside the library.
SMILES_HASH_CCO = "ab1de819ede91df490e6441934decfa9041b49f91d6bbadbba4df634ccff57fa"
#: sha256("100.0000:100.00||||") — content hash of one peak, no context fields.
CONTENT_HASH_ONE_PEAK = "fcc93e405bc0e9689b100c6de77aee904565ca2f0778505da90527cf30b1f37f"

INCHIKEY_ETHANOL = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"


class TestSmilesHash:
    def test_known_vector(self):
        assert smiles_hash("CCO") == SMILES_HASH_CCO
        assert len(SMILES_HASH_CCO) == 64

    def test_is_stable_across_calls(self):
        assert smiles_hash("CCO") == smiles_hash("CCO")

    def test_distinguishes_molecules(self):
        assert smiles_hash("CCO") != smiles_hash("CCN")

    @pytest.mark.parametrize("bad", ["", None, b"CCO"])
    def test_rejects_non_string_or_empty(self, bad):
        with pytest.raises(ValueError):
            smiles_hash(bad)


class TestInchikeySkeleton:
    def test_first_25_characters(self):
        assert inchikey_skeleton(INCHIKEY_ETHANOL) == "LFQSCWFLJHTTHZ-UHFFFAOYSA"
        assert len(inchikey_skeleton(INCHIKEY_ETHANOL)) == 25

    def test_keeps_stereo_block_drops_protonation(self):
        stereo = "LFQSCWFLJHTTHZ-VIFPVBQESA-N"
        assert inchikey_skeleton(stereo) == "LFQSCWFLJHTTHZ-VIFPVBQESA"
        assert inchikey_skeleton(stereo) != inchikey_skeleton(INCHIKEY_ETHANOL)

    def test_rejects_short_value(self):
        with pytest.raises(ValueError):
            inchikey_skeleton("LFQSCWFLJHTTHZ")


class TestContentHash:
    def test_known_vector_single_peak(self):
        assert spectrum_content_hash([100.0], [100.0]) == CONTENT_HASH_ONE_PEAK

    def test_is_peak_order_independent(self):
        a = spectrum_content_hash([100.0, 200.5, 350.25], [100.0, 40.0, 12.5])
        b = spectrum_content_hash([350.25, 100.0, 200.5], [12.5, 100.0, 40.0])
        assert a == b

    def test_context_fields_change_the_key(self):
        base = spectrum_content_hash([100.0], [100.0])
        assert spectrum_content_hash([100.0], [100.0], precursor_mz=101.0) != base
        assert spectrum_content_hash([100.0], [100.0], precursor_type="[M+H]+") != base
        assert spectrum_content_hash([100.0], [100.0], ionization_mode="positive") != base
        assert spectrum_content_hash([100.0], [100.0], collision_energy=30.0) != base

    def test_uses_stored_values_without_renormalising(self):
        # 100/100 and 1/1 give the same rendered peaks only after normalisation;
        # the protocol says "stored values as-is", so the hashes must differ.
        assert spectrum_content_hash([100.0], [100.0]) != spectrum_content_hash([100.0], [1.0])

    def test_intensity_break_tie_for_duplicate_mz(self):
        # duplicate m/z: secondary key is intensity descending, so the pairs are
        # rendered in a deterministic order regardless of input order
        a = spectrum_content_hash([100.0, 100.0], [10.0, 20.0])
        b = spectrum_content_hash([100.0, 100.0], [20.0, 10.0])
        assert a == b


class TestDegeneratePeaks:
    @pytest.mark.parametrize(
        "mz, intensity",
        [
            ([], []),
            ([100.0, 200.0], [100.0]),
            ([100.0], [0.0]),
            ([100.0], [-5.0]),
            ([float("nan")], [1.0]),
            ([100.0], [float("inf")]),
        ],
    )
    def test_degenerate_spectra_are_rejected(self, mz, intensity):
        with pytest.raises(ValueError):
            validate_peaks(mz, intensity)

    def test_degenerate_spectra_cannot_be_hashed(self):
        with pytest.raises(ValueError):
            spectrum_content_hash([], [])


class TestSpectrumKeysAreDeterministic:
    def test_content_hash_keys_are_reproducible(self):
        first = spectra_frame()["spectrum_key"].to_list()
        second = spectra_frame()["spectrum_key"].to_list()
        assert first == second
        assert all(len(key) == 64 for key in first)

    def test_splash_keys_are_carried_through_and_flagged_unverified(self, make_container):
        """C2 scope note: SPLASH is not implemented, so splash keys are opaque."""
        import polars as pl

        from unimol_lite.dataspec import SCHEMA_VERSION, SPEC_VERSION, BuilderInfo, ContainerMeta, HashModes, SourceInfo
        from unimol_lite.dataspec.schema import LINKER_SCHEMA

        spectra = spectra_frame().with_columns(
            pl.Series(
                "spectrum_key",
                [f"splash10-0000-000000000{i}-0000000000" for i in range(3)],
            ),
            pl.Series("hash_mode", ["splash"] * 3),
        )
        bundle = make_container(
            name="splash-passthrough",
            molecules=molecules_frame(),
            spectra=spectra,
            linker=pl.DataFrame(schema=LINKER_SCHEMA),
            meta=ContainerMeta(
                spec_version=SPEC_VERSION,
                schema_version=SCHEMA_VERSION,
                source=SourceInfo(),
                builder=BuilderInfo(name="spectrum-side-tool"),
                hash_modes=HashModes(spectra="splash", molecules="inchikey"),
            ),
        )
        report = validate_container(bundle.root)
        assert report.ok  # unverified keys are a warning, not a protocol error
        assert "spectra.key_unverified" in report.codes()
        assert all(issue.table != TABLE_SPECTRA or issue.severity == "warning" for issue in report.issues)
        assert bundle.container.spectra()["spectrum_key"].to_list()[0].startswith("splash10-")

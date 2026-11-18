"""Fixtures for the SMS-DataSpec adapter suite.

The suite is self-contained (small synthetic containers under ``tmp_path``).  It
needs the optional backend (``polars`` + ``pyarrow``); test modules skip
themselves with ``pytest.importorskip`` when the extra is not installed, so this
conftest only defines the fixtures when the backend is importable.
"""

from __future__ import annotations

import pytest

try:  # the dataspec extra may be absent; the test modules then skip
    from helpers import (
        ContainerBundle,
        container_mode,
        linker_frame,
        molecules_frame,
        spectra_frame,
    )

    from unimol_lite.dataspec import (
        DEFAULT_BLOCK_ROWS,
        SCHEMA_VERSION,
        SPEC_VERSION,
        BuilderInfo,
        ContainerMeta,
        HashModes,
        SourceInfo,
        write_container,
    )
except ImportError:  # pragma: no cover - backend missing
    pass
else:

    @pytest.fixture
    def make_meta():
        """Envelope factory; counts / created_at / fingerprint are finalised on write."""

        def _make(**overrides) -> ContainerMeta:
            fields: dict[str, object] = {
                "spec_version": SPEC_VERSION,
                "schema_version": SCHEMA_VERSION,
                "source": SourceInfo(digest="sha256:" + "0" * 64),
                "builder": BuilderInfo(name="dataspec-tests", version="0.0.0"),
                "hash_modes": HashModes(spectra="content_hash", molecules="inchikey"),
            }
            fields.update(overrides)
            return ContainerMeta(**fields)

        return _make

    @pytest.fixture
    def make_container(tmp_path, make_meta):
        """Build a valid container under ``tmp_path`` (three tables, consistent keys)."""

        def _make(
            name: str = "container",
            *,
            molecules=None,
            spectra=None,
            linker=None,
            meta: ContainerMeta | None = None,
            block_rows: int = DEFAULT_BLOCK_ROWS,
            mixed: bool = False,
            extra_column: bool = False,
        ) -> ContainerBundle:
            root = tmp_path / name
            mols = molecules if molecules is not None else molecules_frame(mixed=mixed, extra_column=extra_column)
            spec = spectra if spectra is not None else spectra_frame()
            link = linker if linker is not None else linker_frame(spec, mols)
            if meta is None:
                meta = make_meta(
                    hash_modes=HashModes(
                        spectra=container_mode(spec["hash_mode"].to_list(), default="content_hash"),
                        molecules=container_mode(mols["key_mode"].to_list(), default="inchikey"),
                    )
                )
            final = write_container(root, mols, spec, link, meta=meta, block_rows=block_rows)
            return ContainerBundle(root=root, meta=final, molecules=mols, spectra=spec, linker=link)

        return _make

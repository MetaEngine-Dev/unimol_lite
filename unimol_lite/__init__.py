"""unimol_lite: lightweight Uni-Mol featurization and model toolkit.

The library is organised in four layers (see ``docs/en/10-solution-layers.md``):

1. **Architecture** — ``unimol_lite.model`` (``UniMolModel`` / ``UniMolConfig`` /
   ``UniMolTokenizer``, with the encoder parts under ``unimol_lite.model.layers``);
2. **Training** — ``unimol_lite.training`` (masked-atom pretraining module,
   datasets, masking) plus the ``unimol-lite pretrain`` CLI; needs the ``train``
   extra;
3. **Standard calls** — this module's ``featurize`` / ``featurize_batch`` /
   ``load_model`` / ``forward_features`` (plus ``unimol_lite.dataspec`` for
   parquet-block container I/O, needing the ``dataspec`` extra);
4. **Minimal use** — ``examples/minimal_features.py`` and the README snippet.

**Responsibility boundary.**  This library produces *molecule features*
(pre-alignment: atom-level encoder states).  Applying a LayerNorm to make a
vector comparable with spectra — the cross-modal alignment step — belongs to the
consuming solution package, not here.  This library does not ship spectra
models, does not train on spectra and does not write alignment vectors.
"""

# ``data`` is imported first on purpose: ``unimol_lite.batch`` imports
# ``.data.prepare_unimol_item``, and ``data/__init__`` re-exports
# ``featurize_batch`` — initializing ``data`` first keeps the cycle one-way.
from . import data  # noqa: F401
from .api import build_model, featurize, featurize_batch, forward_features, load_model
from .model import UniMolConfig, UniMolModel, UniMolTokenizer

__version__ = "0.2.2"

__all__ = [
    # -- layer 1: architecture -------------------------------------------------
    "UniMolModel",
    "UniMolConfig",
    "UniMolTokenizer",
    # -- layer 3: standard calls (layer 2 lives in ``unimol_lite.training``) ---
    "featurize",
    "featurize_batch",
    "build_model",
    "load_model",
    "forward_features",
    # -- package metadata ------------------------------------------------------
    "__version__",
]

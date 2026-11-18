"""Layer 3 — the standard call surface of unimol_lite.

Four functions, one contract each:

``featurize``
    one SMILES -> one model input item (plus the status ``meta`` on demand);
``featurize_batch``
    many SMILES -> one ``(item | None, meta)`` per input, input order preserved;
``build_model`` / ``load_model``
    a randomly initialised model, or a model exported by pretraining;
``forward_features``
    model input item(s) -> ``(encoder_rep, pair_rep)``.

What this layer produces is a **molecule feature** (atom-level encoder states),
i.e. *pre-alignment* features: no LayerNorm is applied here and no spectrum is
involved.  Cross-modal alignment (the ``mol_ln`` step that makes a vector
directly searchable against spectra) belongs to the consuming solution package
and is deliberately not part of this library.

Defaults are aligned with :func:`featurize_batch`: ``timeout=30.0`` for the
single-molecule call as well (the low-level ``smiles2unimolitem`` keeps its
historical ``timeout=None``), and failures are reported the same way —
``return_meta=False`` raises :class:`ValueError`, ``return_meta=True`` returns
``(None, meta)`` with ``meta["status"]``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from .batch import featurize_batch as _featurize_batch

__all__ = [
    "DEFAULT_FEATURIZE_TIMEOUT",
    "ITEM_KEYS",
    "featurize",
    "featurize_batch",
    "build_model",
    "load_model",
    "forward_features",
]

#: Wall-clock budget (seconds) for one conformer, shared by the single and batch calls.
DEFAULT_FEATURIZE_TIMEOUT = 30.0

#: Keys of a model input item, in the order the model expects them.
ITEM_KEYS: tuple[str, ...] = ("src_tokens", "src_coord", "src_distance", "src_edge_type")


def featurize(
    smiles: str,
    *,
    coord_mode: str = "auto",
    timeout: Optional[float] = DEFAULT_FEATURIZE_TIMEOUT,
    on_timeout: str = "fallback_2d",
    rng_seed: int = 42,
    max_atoms: int = 256,
    tokenizer: Any = None,
    return_meta: bool = False,
) -> Union[Dict[str, Any], tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
    """Featurize one SMILES into a model input item.

    Args:
        smiles: molecule to featurize.
        coord_mode: ``"auto"`` (<= 400 atoms -> 3D, else 2D), ``"3d"`` or ``"2d"``.
        timeout: per-conformer budget in seconds (embed + MMFF); ``None`` disables it.
        on_timeout: ``"fallback_2d"`` / ``"error"`` / ``"skip"``.
        rng_seed: seed for conformer choice and atom truncation.
        max_atoms: atom cap; larger molecules are randomly truncated with ``rng_seed``.
        tokenizer: optional shared tokenizer (read-only).
        return_meta: also return the status dict (``status`` / ``coord_mode`` /
            ``reason`` / ``elapsed_s``).

    Returns:
        The item (dict of tensors: ``src_tokens`` ``(L,)`` int64, ``src_coord``
        ``(L, 3)`` float32, ``src_distance`` ``(L, L)`` float32, ``src_edge_type``
        ``(L, L)`` int64), or ``(item | None, meta)`` when ``return_meta=True``.

    Raises:
        ValueError: invalid ``coord_mode`` / ``on_timeout`` / ``timeout``
            parameters, or a molecule that cannot be featurized while
            ``return_meta=False``.
    """
    from .data import smiles2unimolitem

    kwargs: Dict[str, Any] = dict(
        coord_mode=coord_mode,
        timeout=timeout,
        on_timeout=on_timeout,
        rng_seed=rng_seed,
        max_atoms=max_atoms,
        return_meta=True,
    )
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    item, meta = smiles2unimolitem(smiles, **kwargs)
    if return_meta:
        return item, meta
    if item is None:
        raise ValueError(
            f"无法特征化 {smiles!r}: status={meta.get('status')} reason={meta.get('reason')}"
        )
    return item


def featurize_batch(
    smiles: Iterable[str],
    *,
    workers: Optional[int] = None,
    timeout: Optional[float] = DEFAULT_FEATURIZE_TIMEOUT,
    on_timeout: str = "fallback_2d",
    coord_mode: str = "auto",
    rng_seed: int = 42,
    max_atoms: int = 256,
    tokenizer: Any = None,
    window: Optional[int] = None,
) -> List[tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
    """Featurize many SMILES in parallel; one ``(item | None, meta)`` per input.

    Same parameters and defaults as :func:`featurize`; results keep input order
    and input length (duplicates are not folded).  A per-row failure never aborts
    the batch — it is reported through ``meta["status"]``.
    """
    return _featurize_batch(
        smiles,
        workers=workers,
        timeout=timeout,
        on_timeout=on_timeout,
        coord_mode=coord_mode,
        rng_seed=rng_seed,
        max_atoms=max_atoms,
        tokenizer=tokenizer,
        window=window,
    )


def build_model(config: Any = None, *, device: Any = None):
    """Build a model with random weights (demo/structural use, not a trained model).

    Args:
        config: ``UniMolConfig`` instance or a dict of config fields; defaults to
            the reference architecture.
        device: optional torch device to move the model to.
    """
    from .model import UniMolConfig, UniMolModel

    if config is None:
        resolved = UniMolConfig()
    elif isinstance(config, UniMolConfig):
        resolved = config
    elif isinstance(config, Mapping):
        resolved = UniMolConfig(**dict(config))
    else:
        raise ValueError(f"config 必须是 UniMolConfig 或映射, got {type(config).__name__}")
    model = UniMolModel(resolved)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def load_model(path: str, *, device: Any = None):
    """Load a model exported by pretraining (``config.json`` + weights).

    The exported config has ``mode="infer"``, which is what
    :func:`forward_features` requires.
    """
    from .model import UniMolModel

    model = UniMolModel.from_pretrained(str(path))
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def forward_features(model: Any, items: Any, *, device: Any = None):
    """Run the encoder over one item or a list of items; returns feature tensors.

    Args:
        model: a model in inference mode (``build_model`` / ``load_model``, or a
            pretraining export); training-mode configs are refused because they
            return the loss heads instead of the pair representation.
        items: one item dict, or a list of item dicts (padded by the collate
            function, ``pad_to_multiple=8``).
        device: optional device; defaults to the model's parameter device.

    Returns:
        ``(encoder_rep, encoder_pair_rep)``: atom-level features
        ``(B, L, D)`` float32 and pair features ``(B, L, L, heads)`` float32.
        These are *pre-alignment* molecule features — no LayerNorm is applied
        here and no spectrum is involved.

    Raises:
        ValueError: the model is not in inference mode.
    """
    import torch

    from .data.collate_fn import unimol_collate_fn

    configs = getattr(model, "configs", None)
    if getattr(configs, "mode", None) != "infer":
        raise ValueError(
            "forward_features 需要推理态模型（config.mode='infer'）；"
            "训练态模型请用 unimol_lite.training 的损失入口"
        )

    if isinstance(items, Mapping):
        missing = [key for key in ITEM_KEYS if key not in items]
        if missing:
            raise ValueError(f"item 缺键: {missing}")
        batch = {key: items[key].unsqueeze(0) for key in ITEM_KEYS}
    else:
        batch = unimol_collate_fn(list(items))

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            device = None
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            return model(**batch)
    finally:
        if was_training:
            model.train()

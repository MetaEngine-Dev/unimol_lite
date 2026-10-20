# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import zlib

import numpy as np
import contextlib

#: sentinel to tell "argument not passed" from an explicit value
_UNSET = object()


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def derive_seed(seed, *addl_seeds):
    """Stable seed derivation: returns ``seed`` unchanged without ``addl_seeds``,
    otherwise a deterministic CRC32 of ``repr((seed, *addl_seeds))``.

    Unlike ``numpy_seed``'s historical ``hash(...) % 1e6``, this does **not**
    depend on ``PYTHONHASHSEED`` (the old formula was randomized per process
    whenever ``addl_seeds`` contained strings).
    """
    if not addl_seeds:
        return seed
    return zlib.crc32(repr((seed, *addl_seeds)).encode("utf-8"))


def make_rng(seed, *addl_seeds):
    """Return a *local* legacy-style RNG for featurization randomness.

    This is the thread-safe replacement for ``with numpy_seed(...):`` inside
    featurization code: no global state is read or written.

    Without ``addl_seeds`` the seed is used as-is, so sequences are bit-for-bit
    identical to the historical ``np.random.seed(seed)``. With ``addl_seeds``
    the seed is derived by :func:`derive_seed` (process-invariant).

    ``np.random.RandomState`` is required (not ``default_rng``): only the former
    reproduces the historical global-seed sequences bit-for-bit, which keeps
    ``>256`` atom random truncation identical to previous releases
    (see ``_dev/分子采样性能优化/01-目标与设计.md`` §5.2).

    ``seed is None`` keeps the legacy semantics of ``numpy_seed(None)`` — the
    global generator is used as-is (no seeding, hence no reproducibility).
    """
    if seed is None:
        return np.random
    return np.random.RandomState(derive_seed(seed, *addl_seeds))

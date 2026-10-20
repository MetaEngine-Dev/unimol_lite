"""批次 1：局部 RNG 与旧全局 seed 序列的逐位兼容性。"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from unimol_lite.data.utils import derive_seed, make_rng, numpy_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_randomstate_matches_legacy_global_choice():
    np.random.seed(42)
    expected = np.random.choice(300, 256, replace=False)
    rng = np.random.RandomState(42)
    assert np.array_equal(rng.choice(300, 256, replace=False), expected)


def test_randomstate_matches_legacy_global_randint():
    np.random.seed(42)
    expected = np.random.randint(3, size=10)
    rng = np.random.RandomState(42)
    assert np.array_equal(rng.randint(3, size=10), expected)


def test_make_rng_matches_numpy_seed():
    with numpy_seed(42):
        expected = np.random.choice(50, 10, replace=False)
    assert np.array_equal(make_rng(42).choice(50, 10, replace=False), expected)


def test_derive_seed_without_addl_seeds_is_identity():
    assert derive_seed(42) == 42
    assert derive_seed(0) == 0


def test_derive_seed_with_addl_seeds_is_deterministic():
    first = derive_seed(42, "fold")
    second = derive_seed(42, "fold")
    assert first == second
    assert first != derive_seed(42, "other")
    # same inputs -> same RNG stream
    assert np.array_equal(
        make_rng(42, "fold").randint(100, size=8),
        make_rng(42, "fold").randint(100, size=8),
    )


def test_make_rng_does_not_touch_global_state():
    np.random.seed(7)
    before = np.random.get_state()
    rng = make_rng(42)
    rng.choice(50, 10, replace=False)
    after = np.random.get_state()
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]


def test_make_rng_unseeded_follows_global_state():
    np.random.seed(11)
    expected = np.random.randint(100, size=4)
    np.random.seed(11)
    assert np.array_equal(make_rng(None).randint(100, size=4), expected)


def test_derive_seed_stable_across_python_hash_seeds():
    code = (
        "from unimol_lite.data.utils import make_rng;"
        "print(make_rng(42, 'fold').randint(1000, size=8).tolist())"
    )
    outputs = []
    for hash_seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, text=True
            ).strip()
        )
    assert outputs[0] == outputs[1]

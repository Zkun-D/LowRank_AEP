import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adaptive_basis import select_adaptive_basis_strategy_A


def test_nested_pivot_preserving_basis():
    rng = np.random.default_rng(20260720)
    wd = np.arange(0.0, 360.0, 5.0)
    for _ in range(200):
        probe = rng.uniform(0.0, 1.0, 72)
        energy = rng.uniform(0.0, 1.0, 72)
        pivot = int(rng.integers(0, 72))
        sets = []
        for k in range(1, 7):
            basis, *_ = select_adaptive_basis_strategy_A(
                probe, energy, pivot, wd, k=k, fill_strategy="maximin"
            )
            assert pivot in basis
            assert len(np.unique(basis)) == k
            sets.append(set(basis.tolist()))
        for left, right in zip(sets[:-1], sets[1:]):
            assert left.issubset(right)


def test_k1_is_pivot_only():
    probe = np.linspace(0.2, 0.9, 72)
    energy = np.ones(72)
    basis, *_ = select_adaptive_basis_strategy_A(
        probe, energy, 17, np.arange(0.0, 360.0, 5.0), k=1
    )
    assert basis.tolist() == [17]

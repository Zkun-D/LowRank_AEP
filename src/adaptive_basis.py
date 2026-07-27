"""Adaptive basis selection used by the public low-rank evaluator.

The hierarchy is nested and pivot preserving:
K=1: pivot only (Rank-1)
K=2: pivot + farther probe-space extreme
K=3: pivot + both extremes
K=4: pivot + both extremes + Strategy-A intermediate direction
K>4: maximin or energy-based fill

For K >= 4, the returned row order preserves the submitted paper-default
[min, intermediate, max, pivot] ordering when those roles are distinct.  This
keeps plateau tie handling consistent with the reference calculations while
the selected basis set remains nested.
"""
from __future__ import annotations

import numpy as np


def angular_distance_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def _append_unique(sequence: list[int], index: int) -> None:
    index = int(index)
    if index not in sequence:
        sequence.append(index)


def _pick_extremes(
    probe: np.ndarray,
    energy: np.ndarray,
    wd_array: np.ndarray,
    pivot_idx: int,
    atol_x_max: float,
) -> tuple[int, int]:
    x = np.asarray(probe, dtype=float)
    e = np.asarray(energy, dtype=float)
    wd = np.asarray(wd_array, dtype=float)
    pivot_idx = int(pivot_idx)

    idx_min = int(np.argmin(x))
    xmax = float(np.max(x))
    candidates = np.where(x >= xmax - float(atol_x_max))[0]
    if candidates.size == 0:
        return idx_min, int(np.argmax(x))

    max_energy = np.max(e[candidates])
    candidates = candidates[e[candidates] == max_energy]
    if candidates.size == 1:
        idx_max = int(candidates[0])
    else:
        distances = np.array(
            [angular_distance_deg(wd[i], wd[pivot_idx]) for i in candidates],
            dtype=float,
        )
        idx_max = int(candidates[np.argmax(distances)])
    return idx_min, idx_max


def _midpoint_strategy_a(
    probe: np.ndarray,
    energy: np.ndarray,
    idx_min: int,
    idx_max: int,
) -> int:
    x = np.asarray(probe, dtype=float)
    e = np.asarray(energy, dtype=float)
    score = np.minimum(np.abs(x - x[idx_min]), np.abs(x - x[idx_max]))
    candidates = np.where(score == np.max(score))[0]
    if candidates.size == 1:
        return int(candidates[0])
    return int(candidates[np.argmax(e[candidates])])


def _ordered_extremes_from_pivot(
    probe: np.ndarray,
    energy: np.ndarray,
    pivot_idx: int,
    idx_min: int,
    idx_max: int,
) -> list[int]:
    x = np.asarray(probe, dtype=float)
    e = np.asarray(energy, dtype=float)
    candidates: list[int] = []
    for index in (int(idx_min), int(idx_max)):
        _append_unique(candidates, index)
    pivot_value = float(x[int(pivot_idx)])
    return sorted(
        candidates,
        key=lambda index: (abs(float(x[index]) - pivot_value), float(e[index])),
        reverse=True,
    )


def _fill_basis(
    basis: list[int],
    probe: np.ndarray,
    energy: np.ndarray,
    target_k: int,
    fill_strategy: str,
) -> list[int]:
    x = np.asarray(probe, dtype=float)
    e = np.asarray(energy, dtype=float)
    remaining = [index for index in range(len(x)) if index not in basis]
    strategy = str(fill_strategy).lower()

    if strategy == "energy":
        if remaining:
            arr = np.asarray(remaining, dtype=int)
            for index in arr[np.argsort(-e[arr])]:
                _append_unique(basis, int(index))
                if len(basis) >= target_k:
                    break
    elif strategy == "maximin":
        while len(basis) < target_k and remaining:
            arr = np.asarray(remaining, dtype=int)
            basis_values = x[np.asarray(basis, dtype=int)]
            distances = np.min(np.abs(x[arr, None] - basis_values[None, :]), axis=1)
            candidates = arr[distances == np.max(distances)]
            if candidates.size == 1:
                pick = int(candidates[0])
            else:
                pick = int(candidates[np.argmax(e[candidates])])
            _append_unique(basis, pick)
            remaining.remove(pick)
    else:
        raise ValueError("fill_strategy must be 'maximin' or 'energy'")
    return basis


def select_adaptive_basis_strategy_A(
    probe,
    energy_per_wd,
    pivot_idx,
    wd_array,
    k: int = 4,
    atol_x_max: float = 1e-6,
    fill_strategy: str = "maximin",
):
    """Return the nested Strategy-A basis and its min/mid/max roles."""
    x = np.asarray(probe, dtype=float)
    e = np.asarray(energy_per_wd, dtype=float)
    pivot = int(pivot_idx)
    target_k = int(k)

    if target_k < 1:
        raise ValueError("adaptive_k must be at least 1")
    if target_k > len(x):
        raise ValueError("adaptive_k exceeds the number of wind directions")

    idx_min, idx_max = _pick_extremes(
        x, e, np.asarray(wd_array, dtype=float), pivot, atol_x_max
    )
    idx_mid = _midpoint_strategy_a(x, e, idx_min, idx_max)
    ordered_extremes = _ordered_extremes_from_pivot(
        x, e, pivot, idx_min, idx_max
    )

    priority = [pivot, *ordered_extremes, idx_mid]
    basis: list[int] = []
    for index in priority:
        _append_unique(basis, int(index))
        if len(basis) >= target_k:
            break

    if len(basis) < target_k:
        basis = _fill_basis(basis, x, e, target_k, fill_strategy)

    if pivot not in basis or len(basis) != target_k:
        raise RuntimeError("Adaptive basis construction failed")

    # Preserve the submitted K>=4 row order for exact default reproduction.
    if target_k >= 4:
        ordered_output: list[int] = []
        for index in (idx_min, idx_mid, idx_max, pivot):
            if int(index) in basis:
                _append_unique(ordered_output, int(index))
        for index in basis:
            _append_unique(ordered_output, int(index))
        basis = ordered_output[:target_k]

    return (
        np.asarray(basis, dtype=int),
        int(idx_min),
        int(idx_mid),
        int(idx_max),
    )

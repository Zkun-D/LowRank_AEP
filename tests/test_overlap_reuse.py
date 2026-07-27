"""Check that Stage-I overlap reuse preserves AEP and reduces raw queries."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_evaluator_with_fake_floris(power_matrix):
    floris = types.ModuleType("floris")

    class WindRose:
        def __init__(self, wind_directions, wind_speeds, ti_table, freq_table):
            self.freq_table = np.asarray(freq_table)

    class FlorisModel:
        def __init__(self, _):
            self.wind_data = None

        def set(self, layout_x=None, layout_y=None, wind_data=None):
            if wind_data is not None:
                self.wind_data = wind_data

        def run(self):
            return None

        def get_farm_power(self):
            return power_matrix * (self.wind_data.freq_table > 0)

    floris.FlorisModel = FlorisModel
    floris.WindRose = WindRose

    fake_names = [
        "floris",
        "floris.optimization",
        "floris.optimization.layout_optimization",
        "floris.optimization.layout_optimization.layout_optimization_base",
    ]
    saved = {name: sys.modules.get(name) for name in fake_names}
    sys.modules["floris"] = floris
    sys.modules["floris.optimization"] = types.ModuleType("floris.optimization")
    sys.modules["floris.optimization.layout_optimization"] = types.ModuleType(
        "floris.optimization.layout_optimization"
    )
    base = types.ModuleType(
        "floris.optimization.layout_optimization.layout_optimization_base"
    )
    base.LayoutOptimization = type("LayoutOptimization", (), {})
    sys.modules[
        "floris.optimization.layout_optimization.layout_optimization_base"
    ] = base

    sys.path.insert(0, str(ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location(
            "lowrank_optimizer_test", ROOT / "src" / "lowrank_optimizer.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.compute_aep_with_evaluator
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_overlap_reuse_preserves_result_and_reduces_raw_calls():
    n_wd, n_ws = 72, 25
    wd = np.arange(0.0, 360.0, 5.0)
    ws = np.arange(0.5, 25.0, 1.0)
    p_no_wake = np.minimum((ws / 12.5) ** 3, 1.0) * 36.0e6
    directional = 0.76 + 0.20 * (1.0 + np.cos(np.deg2rad(wd - 35.0))) / 2.0
    speed_factor = 0.88 + 0.10 * np.minimum(ws / 12.5, 1.0)
    power = np.outer(directional, p_no_wake * speed_factor)

    freq = np.ones((n_wd, n_ws), dtype=float)
    freq /= freq.sum()
    ti = np.full_like(freq, 0.06)
    evaluator = _load_evaluator_with_fake_floris(power)

    common = dict(
        layout_x=[0.0],
        layout_y=[0.0],
        fmodel_dict={},
        wind_directions=wd,
        wind_speeds=ws,
        ti_table=ti,
        freq_table=freq,
        p_no_wake_1d=p_no_wake,
        lowrank_cut_in=3.0,
        eff_unity_ws_start=17.0,
        mode="adaptive",
        precomputed_pivots={"u_opt_idx": 10, "v_opt_idx": 7},
        effective_n_ws=n_ws,
        force_low_speed_correction=False,
        adaptive_k=4,
        adaptive_fill_strategy="maximin",
        return_diagnostics=True,
    )
    aep_reuse, diag_reuse = evaluator(**common, reuse_stage1_overlap=True)
    aep_repeat, diag_repeat = evaluator(**common, reuse_stage1_overlap=False)

    assert np.isclose(aep_reuse, aep_repeat, rtol=0.0, atol=1e-6)
    assert diag_reuse["unique_query_count"] == diag_repeat["unique_query_count"]
    assert diag_reuse["raw_query_count"] < diag_repeat["raw_query_count"]
    assert diag_reuse["reused_query_count"] > 0

"""
Example script for running one low-rank WFLO case.

Usage
-----
python examples/run_single_case.py --config examples/parameters_example.txt

Optional override
-----------------
python examples/run_single_case.py --config examples/parameters_example.txt --mode exact
python examples/run_single_case.py --config examples/parameters_example.txt --mode rank1
python examples/run_single_case.py --config examples/parameters_example.txt --mode adaptive
python examples/run_single_case.py --config examples/parameters_example.txt --mode hybrid
"""

from __future__ import annotations
import os
import sys
import argparse
import pickle
import traceback
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from floris import FlorisModel
from config_utils import load_config
from case_utils import (
    load_wind_resource,
    apply_cut_out,
    load_ti_table,
    build_wind_rose,
    build_scaled_boundary,
    generate_initial_layout,
)
from lowrank_optimizer import LayoutOptimizationRandomSearchLowRank


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single low-rank WFLO case.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to parameter file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Optional evaluator override: exact / rank1 / adaptive / hybrid",
    )
    return parser.parse_args()


def _infer_history_layout_keys(history):
    """
    Infer layout key names used in optimizer history.

    Supported variants:
    - layouts_x / layouts_y   (current)
    - layout_x / layout_y
    - layouts / layouts_y     (legacy)
    """
    if not history:
        return None, None

    for item in history:
        if not isinstance(item, dict):
            continue

        if ("layouts_x" in item) and ("layouts_y" in item):
            return "layouts_x", "layouts_y"

        if ("layout_x" in item) and ("layout_y" in item):
            return "layout_x", "layout_y"

        if ("layouts" in item) and ("layouts_y" in item):
            return "layouts", "layouts_y"

    return None, None


def main():
    args = parse_args()

    try:
        config = load_config(args.config)
        if args.mode is not None:
            config["evaluator"]["mode"] = args.mode.strip().lower()
            if config["evaluator"]["mode"] == "full":
                config["evaluator"]["mode"] = "exact"
    except Exception as e:
        print(f"[CRITICAL] Failed to load configuration: {e}")
        raise

    floris_input_file = config["floris"]["input_file"]
    case_cfg = config["case"]
    layout_cfg = config["layout"]
    opt_cfg = config["optimization"]
    eval_cfg = config["evaluator"]
    wind_cfg = config["wind"]
    out_cfg = config["output"]

    case_name = case_cfg["case_name"]
    aep_mode = eval_cfg["mode"]

    print("=" * 72)
    print("Low-Rank WFLO Example Run")
    print("=" * 72)
    print(f"Case name         : {case_name}")
    print(f"Case type         : {case_cfg['case_type']}")
    print(f"Evaluator mode    : {aep_mode}")
    print(f"FLORIS input file : {floris_input_file}")
    print("=" * 72)

    try:
        # -------------------------
        # Load wind resource
        # -------------------------
        wind_directions, wind_speeds, freq_table, freq_source = load_wind_resource(case_cfg)
        freq_table = apply_cut_out(
            freq_table=freq_table,
            wind_speeds=wind_speeds,
            cut_out_ws=wind_cfg["cut_out_ws"],
            use_cut_out=wind_cfg["use_cut_out"],
        )

        ti_table, ti_source = load_ti_table(
            case_config=case_cfg,
            wind_config=wind_cfg,
            freq_table_shape=freq_table.shape,
        )

        wind_rose = build_wind_rose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            freq_table=freq_table,
            ti_table=ti_table,
        )

        print(f"[Wind] freq source: {freq_source}")
        print(f"[Wind] TI source  : {ti_source}")
        print(f"[Wind] freq shape : {freq_table.shape}")
        print(f"[Wind] TI range   : [{np.nanmin(ti_table):.4f}, {np.nanmax(ti_table):.4f}]")

        # -------------------------
        # Initialize FLORIS model
        # -------------------------
        fmodel = FlorisModel(floris_input_file)
        fmodel.set(wind_data=wind_rose)

        D = float(fmodel.core.farm.rotor_diameters.max())

        Nx = layout_cfg["Nx"]
        Ny = layout_cfg["Ny"]
        spacing_D = layout_cfg["spacing_D"]
        min_spacing = layout_cfg["min_spacing_D"] * D
        n_turbines = Nx * Ny

        boundaries, target_area, scaled_area = build_scaled_boundary(
            Nx=Nx,
            Ny=Ny,
            spacing_D=spacing_D,
            shape=layout_cfg["shape"],
            aspect_ratio=layout_cfg["aspect_ratio"],
            rotor_diameter=D,
        )

        print(f"[Layout] Turbines         : {n_turbines} ({Nx}x{Ny})")
        print(f"[Layout] Spacing target   : {spacing_D[0]:.2f}D x {spacing_D[1]:.2f}D")
        print(f"[Layout] Minimum spacing  : {layout_cfg['min_spacing_D']:.2f}D")
        print(f"[Layout] Boundary shape   : {layout_cfg['shape']}")
        print(f"[Layout] Target area      : {target_area:.3f}")
        print(f"[Layout] Scaled area      : {scaled_area:.3f}")

        # -------------------------
        # Generate one feasible initial layout for fmodel
        # -------------------------
        print("[Init] Generating feasible initial layout ...")
        initial_layout = generate_initial_layout(
            n_turbines=n_turbines,
            boundary_vertices=boundaries,
            min_spacing=min_spacing,
            seed=opt_cfg["random_seed"],
        )
        fmodel.set(layout_x=initial_layout[:, 0], layout_y=initial_layout[:, 1])
        print("[Init] Done.")

        # -------------------------
        # Initialize optimizer
        # -------------------------
        optimizer = LayoutOptimizationRandomSearchLowRank(
            fmodel=fmodel,
            boundaries=boundaries,
            min_dist=min_spacing,
            n_individuals=opt_cfg["n_individuals"],
            seconds_per_iteration=opt_cfg["seconds_per_iteration"],
            total_optimization_seconds=opt_cfg["total_optimization_seconds"],
            interface=opt_cfg["interface"],
            max_workers=opt_cfg["n_workers"],
            grid_step_size=opt_cfg["grid_step_size"],
            relegation_number=opt_cfg["relegation_number"],
            random_seed=opt_cfg["random_seed"],
            aep_mode=aep_mode,
            hybrid_exact_start_ratio=eval_cfg["hybrid_exact_start_ratio"],
            lowrank_cut_in=eval_cfg["lowrank_cut_in"],
            rated_ws=eval_cfg["rated_ws"],
            eff_unity_ws_start=eval_cfg["eff_unity_ws_start"],
            pivot_ws_min=eval_cfg["pivot_ws_min"],
            pivot_ws_max=eval_cfg["pivot_ws_max"],
            adaptive_k=eval_cfg["adaptive_k"],
            tie_atol_x_max=eval_cfg["tie_atol_x_max"],
        )

        optimizer.describe()

        # -------------------------
        # Run optimization
        # -------------------------
        print("\n[Run] Starting optimization ...")
        objective_final, x_opt, y_opt = optimizer.optimize()

        history_layout_x_key, history_layout_y_key = _infer_history_layout_keys(optimizer.history)
        if out_cfg["save_history"]:
            print(f"[History] layout keys: {history_layout_x_key}, {history_layout_y_key}")

        # -------------------------
        # Save results
        # -------------------------
        results_dir = out_cfg["results_dir"]
        os.makedirs(results_dir, exist_ok=True)

        result_filename = f"{case_name}_{aep_mode}_result.pkl"
        result_path = os.path.join(results_dir, result_filename)

        result = {
            "x_opt": x_opt,
            "y_opt": y_opt,
            "objective_initial": optimizer.objective_initial,
            "objective_final": objective_final,
            "history": optimizer.history if out_cfg["save_history"] else None,
            "history_layout_x_key": history_layout_x_key,
            "history_layout_y_key": history_layout_y_key,
            "num_objective_calls_log": getattr(optimizer, "num_objective_calls_log", None),
            "num_accept_log": getattr(optimizer, "num_accept_log", None),
            "total_objective_calls": getattr(optimizer, "total_objective_calls", None),
            "eval_mode_log": getattr(optimizer, "eval_mode_log", None),
            "wind_data_info": {
                "freq_source": freq_source,
                "ti_source": ti_source,
                "ti_stats": {
                    "min": float(np.nanmin(ti_table)),
                    "max": float(np.nanmax(ti_table)),
                    "mean": float(np.nanmean(ti_table)),
                },
                "wind_speeds": wind_speeds,
                "wind_directions": wind_directions,
            },
            "config": config,
        }

        with open(result_path, "wb") as f:
            pickle.dump(result, f)

        print("\n" + "=" * 72)
        print("Run completed successfully.")
        print(f"Final objective : {objective_final / 1e9:.6f} GWh")
        print(f"Saved result to : {result_path}")
        print("=" * 72)

    except Exception as e:
        print("\n[ERROR] Run failed.")
        print(f"Reason: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
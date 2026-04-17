"""
Configuration utilities for low-rank WFLO.

This module reads a simple key=value text configuration file and converts
entries into a structured dictionary with validated types and defaults.
"""

from __future__ import annotations
import os


def read_parameter_file(file_path: str) -> dict:
    """
    Read a text parameter file with format:

        key = value

    Lines starting with '#' or content after '#' are ignored.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parameter file not found: {file_path}")

    raw = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            raw[key.strip()] = value.strip()

    return raw


def _parse_bool_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _parse_optional_int(value, default=None):
    if value is None:
        return default
    s = str(value).strip()
    if s.lower() == "none":
        return default
    return int(s)


def _parse_optional_float(value, default=None):
    if value is None:
        return default
    s = str(value).strip()
    if s.lower() == "none":
        return default
    return float(s)


def _parse_str(value, default=None):
    if value is None:
        return default
    return str(value).strip().strip("'").strip('"')


def _parse_optional_str(value, default=None):
    if value is None:
        return default
    s = str(value).strip().strip("'").strip('"')
    if s.lower() == "none":
        return default
    return s


def _parse_float_list(value):
    s = str(value).strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected list format like [a, b], got: {value}")
    body = s[1:-1].strip()
    if not body:
        return []
    return [float(x.strip()) for x in body.split(",")]


def _normalize_mode(raw_mode: str) -> str:
    mode = _parse_str(raw_mode, default="rank1").lower()
    mode_map = {
        "exact": "exact",
        "full": "exact",
        "full_floris": "exact",
        "rank1": "rank1",
        "adaptive": "adaptive",
        "hybrid": "hybrid",
        "hybrid_rank1_full": "hybrid",
    }
    if mode not in mode_map:
        raise ValueError(
            f"Invalid aep_mode '{raw_mode}'. Valid options are: "
            f"{list(sorted(mode_map.keys()))}"
        )
    return mode_map[mode]


def _resolve_path(path_value, base_dir, default=None):
    """
    Resolve a possibly relative path against the parameter-file directory.
    """
    path_str = _parse_optional_str(path_value, default=default)
    if path_str is None:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(base_dir, path_str))


def load_config(file_path: str) -> dict:
    """
    Load, parse, validate, and structure configuration parameters.
    """
    raw = read_parameter_file(file_path)
    config_dir = os.path.dirname(os.path.abspath(file_path))

    floris_input_file = _resolve_path(
        raw.get("floris_input_file", "floris_inputs/gch.yaml"),
        base_dir=config_dir,
    )

    wind_rose_file = _resolve_path(
        raw.get("wind_rose_file", None),
        base_dir=config_dir,
        default=None,
    )

    ti_file = _resolve_path(
        raw.get("ti_file", None),
        base_dir=config_dir,
        default=None,
    )

    results_dir = _resolve_path(
        raw.get("results_dir", "results"),
        base_dir=config_dir,
    )

    config = {
        "floris": {
            "input_file": floris_input_file,
        },
        "case": {
            "case_name": _parse_str(raw.get("case_name", "example_case")),
            "case_type": _parse_str(raw.get("case_type", "real")).lower(),
            "case_index": int(raw.get("case_index", 1)),
            "wind_rose_file": wind_rose_file,
            "ti_file": ti_file,
        },
        "layout": {
            "Nx": int(raw.get("Nx", 4)),
            "Ny": int(raw.get("Ny", 4)),
            "spacing_D": _parse_float_list(raw.get("spacing_D", "[6.0, 6.0]")),
            "min_spacing_D": float(raw.get("min_spacing_D", 3.0)),
            "shape": _parse_str(raw.get("shape", "Rectangle")),
            "aspect_ratio": float(raw.get("aspect_ratio", 1.0)),
        },
        "optimization": {
            "n_individuals": int(raw.get("n_individuals", 4)),
            "seconds_per_iteration": float(raw.get("seconds_per_iteration", 60.0)),
            "total_optimization_seconds": float(raw.get("total_optimization_seconds", 600.0)),
            "n_workers": int(raw.get("n_workers", raw.get("n_works", 1))),
            "random_seed": _parse_optional_int(raw.get("random_seed", None), default=None),
            "grid_step_size": float(raw.get("grid_step_size", 50.0)),
            "relegation_number": int(raw.get("relegation_number", 1)),
            "interface": _parse_optional_str(raw.get("interface", "multiprocessing"), default="multiprocessing"),
        },
        "evaluator": {
            "mode": _normalize_mode(raw.get("aep_mode", "rank1")),
            "hybrid_exact_start_ratio": float(raw.get("hybrid_exact_start_ratio", raw.get("hybrid_full_start_ratio", 0.5))),
            "lowrank_cut_in": float(raw.get("lowrank_cut_in", 3.0)),
            "rated_ws": float(raw.get("rated_ws", 12.5)),
            "eff_unity_ws_start": float(raw.get("eff_unity_ws_start", 17.0)),
            "pivot_ws_min": float(raw.get("pivot_ws_min", 4.0)),
            "pivot_ws_max": float(raw.get("pivot_ws_max", 12.5)),
            "adaptive_k": int(raw.get("adaptive_k", 4)),
            "tie_atol_x_max": float(raw.get("tie_atol_x_max", 1e-6)),
        },
        "wind": {
            "use_cut_out": _parse_bool_flag(raw.get("use_cut_out", 1), default=True),
            "cut_out_ws": float(raw.get("cut_out_ws", 25.0)),
            "ti_constant": float(raw.get("ti_constant", 0.06)),
        },
        "output": {
            "results_dir": results_dir,
            "save_history": _parse_bool_flag(raw.get("save_history", 1), default=True),
        },
    }

    if len(config["layout"]["spacing_D"]) != 2:
        raise ValueError("spacing_D must contain exactly two values, e.g. [6.0, 6.0].")

    ratio = config["evaluator"]["hybrid_exact_start_ratio"]
    if not (0.0 <= ratio <= 1.0):
        raise ValueError("hybrid_exact_start_ratio must be in [0, 1].")

    if config["evaluator"]["pivot_ws_max"] <= config["evaluator"]["pivot_ws_min"]:
        raise ValueError("pivot_ws_max must be greater than pivot_ws_min.")

    shape = config["layout"]["shape"]
    if shape not in {"Circle", "Rectangle"}:
        raise ValueError("shape must be either 'Circle' or 'Rectangle'.")

    case_type = config["case"]["case_type"]
    if case_type not in {"real", "classical"}:
        raise ValueError("case_type must be either 'real' or 'classical'.")

    if not os.path.exists(config["floris"]["input_file"]):
        raise FileNotFoundError(
            f"FLORIS input file not found: {config['floris']['input_file']}"
        )

    if case_type == "real":
        if config["case"]["wind_rose_file"] is None:
            default_wr = os.path.normpath(
                os.path.join(config_dir, f"wind_conditions/windRose_{config['case']['case_index']}.npy")
            )
            config["case"]["wind_rose_file"] = default_wr

        if not os.path.exists(config["case"]["wind_rose_file"]):
            raise FileNotFoundError(
                f"Wind resource file not found: {config['case']['wind_rose_file']}"
            )

    if config["case"]["ti_file"] is not None:
        if not os.path.exists(config["case"]["ti_file"]):
            raise FileNotFoundError(
                f"TI file not found: {config['case']['ti_file']}"
            )

    return config
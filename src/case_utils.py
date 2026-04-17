"""
Case construction utilities for low-rank WFLO.

This module handles:
- wind resource loading
- TI table loading
- cut-out processing
- boundary construction and scaling
- simple random initial layout generation
"""

from __future__ import annotations
import os
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial.distance import cdist
from floris import WindRose


def load_wind_resource(case_config: dict):
    """
    Load wind-resource frequency table and wind-speed array.

    Supported case types:
    - real
    - classical
    """
    case_type = case_config["case_type"]
    case_index = case_config["case_index"]
    wind_rose_file = case_config.get("wind_rose_file", None)

    wind_directions = np.arange(0, 360, 5) - 2.5

    if case_type == "real":
        if wind_rose_file is None:
            raise FileNotFoundError(
                "For real cases, 'wind_rose_file' must be provided in config "
                "or resolved before calling load_wind_resource()."
            )
        if not os.path.exists(wind_rose_file):
            raise FileNotFoundError(f"Wind resource file not found: {wind_rose_file}")

        freq_table = np.load(wind_rose_file)
        wind_speeds = np.arange(0.5, 25.0, 1.0)

    elif case_type == "classical":
        if case_index == 1:
            wind_speeds = np.array([12.0])
            freq_table = np.zeros((len(wind_directions), len(wind_speeds)))
            freq_table[0, 0] = 1.0
        elif case_index == 2:
            wind_speeds = np.array([12.0])
            freq_table = np.ones((len(wind_directions), len(wind_speeds)))
            freq_table /= freq_table.sum()
        else:
            raise ValueError(f"Unknown classical case index: {case_index}")
    else:
        raise ValueError(f"Unsupported case type: {case_type}")

    return wind_directions, wind_speeds, freq_table, wind_rose_file


def apply_cut_out(freq_table, wind_speeds, cut_out_ws=25.0, use_cut_out=True):
    """
    Zero out frequencies above cut-out wind speed and renormalize.
    """
    freq_table = np.asarray(freq_table, dtype=float).copy()

    if not use_cut_out:
        return freq_table

    cut_mask = np.asarray(wind_speeds) > float(cut_out_ws)
    if np.any(cut_mask):
        freq_table[:, cut_mask] = 0.0
        s = np.sum(freq_table)
        if s > 0:
            freq_table /= s

    return freq_table


def load_ti_table(case_config: dict, wind_config: dict, freq_table_shape):
    """
    Load case-specific TI matrix if available; otherwise use constant TI.
    """
    ti_file = case_config.get("ti_file", None)

    if ti_file is not None and os.path.exists(ti_file):
        ti_table = np.load(ti_file).astype(float)
        if ti_table.shape != freq_table_shape:
            raise ValueError(
                f"TI matrix shape mismatch: ti_table.shape={ti_table.shape}, "
                f"expected={freq_table_shape}, file={ti_file}"
            )
        ti_source = ti_file
    else:
        ti_const = float(wind_config["ti_constant"])
        ti_table = ti_const * np.ones(freq_table_shape, dtype=float)
        ti_source = f"constant:{ti_const:.4f}"

    return ti_table, ti_source


def build_wind_rose(wind_directions, wind_speeds, freq_table, ti_table):
    """
    Construct FLORIS WindRose object.
    """
    value_table = np.ones_like(freq_table, dtype=float)
    return WindRose(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        ti_table=ti_table,
        freq_table=freq_table,
        value_table=value_table,
    )


def build_scaled_boundary(Nx, Ny, spacing_D, shape="Rectangle", aspect_ratio=1.0, rotor_diameter=126.0):
    """
    Build and scale a simple parametric boundary to match the target layout area.
    """
    if shape == "Circle":
        theta = np.linspace(0, 2 * np.pi, 500)
        initial_boundary = np.column_stack((np.cos(theta), np.sin(theta)))
    elif shape == "Rectangle":
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1]) * aspect_ratio
        initial_boundary = np.column_stack((x, y))
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    target_width = (Nx - 1) * spacing_D[0] * rotor_diameter
    target_height = (Ny - 1) * spacing_D[1] * rotor_diameter
    target_area = target_width * target_height

    x_coords = np.array([p[0] for p in initial_boundary])
    y_coords = np.array([p[1] for p in initial_boundary])

    current_area = 0.5 * np.abs(
        np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1))
    )
    if current_area < 1e-12:
        raise ValueError("Initial boundary has near-zero area.")

    scale_factor = np.sqrt(target_area / current_area)
    centroid_x, centroid_y = np.mean(x_coords), np.mean(y_coords)

    scaled_boundary = [
        (
            centroid_x + scale_factor * (x - centroid_x),
            centroid_y + scale_factor * (y - centroid_y),
        )
        for x, y in initial_boundary
    ]

    if scaled_boundary[0] != scaled_boundary[-1]:
        scaled_boundary.append(scaled_boundary[0])

    sx = np.array([p[0] for p in scaled_boundary])
    sy = np.array([p[1] for p in scaled_boundary])
    scaled_area = 0.5 * np.abs(np.dot(sx, np.roll(sy, 1)) - np.dot(sy, np.roll(sx, 1)))

    return scaled_boundary, target_area, scaled_area


def generate_initial_layout(n_turbines, boundary_vertices, min_spacing, seed=None):
    """
    Generate a random initial layout within the boundary while satisfying
    minimum spacing constraints.
    """
    if seed is not None:
        np.random.seed(seed)

    boundary_polygon = Polygon(boundary_vertices)
    turbine_positions = []

    max_attempts = n_turbines * 1000
    attempts = 0

    min_x, min_y, max_x, max_y = boundary_polygon.bounds

    while len(turbine_positions) < n_turbines:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not place {n_turbines} turbines with spacing {min_spacing:.3f}."
            )

        candidate = (
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
        )

        if not boundary_polygon.contains(Point(candidate)):
            continue

        if turbine_positions:
            distances = cdist([candidate], turbine_positions)
            if np.all(distances >= min_spacing):
                turbine_positions.append(candidate)
        else:
            turbine_positions.append(candidate)

    return np.array(turbine_positions)
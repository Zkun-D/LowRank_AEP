# Low-Rank Wind Farm Layout Optimization

A FLORIS-based framework for wind farm layout optimization (WFLO) with accelerated annual energy production (AEP) evaluation using low-rank reconstruction.

This repository supports four evaluator modes:

- `exact`
- `rank1`
- `adaptive`
- `hybrid`

The code is designed for research on fast wake-aware layout optimization under multiple wind directions and wind speeds.

---

## Overview

Wind farm layout optimization is computationally expensive because each candidate layout must be evaluated over many wind directions (WD) and wind speeds (WS). With a high-resolution wind rose, wake-model calls quickly become the dominant cost.

This repository implements a low-rank acceleration framework on top of FLORIS:

- **exact**: direct wake-model evaluation on the supplied WD-WS support
- **rank1**: rank-1 reconstruction from sparse cross sampling
- **adaptive**: adaptive sparse reconstruction using the final paper-aligned basis strategy
- **hybrid**: rank1 in the early stage of optimization and exact in the later stage

The framework is intended for methodological studies on accelerating wake-aware layout optimization while preserving consistency with FLORIS-based physics evaluation.

---

## Features

- FLORIS-based wake-aware layout optimization
- Low-rank AEP evaluation for faster optimization
- Four evaluator modes: `exact`, `rank1`, `adaptive`, and `hybrid`
- Real and idealized wind-resource cases
- Optional loading of turbulence-intensity (TI) matrices
- Random-search optimization with population-based evolution
- Automatic result saving with optimization history and evaluator logs

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src
│   ├── __init__.py
│   ├── config_utils.py
│   ├── case_utils.py
│   └── lowrank_optimizer.py
├── examples
│   ├── run_single_case.py
│   └── parameters_example.txt
├── floris_inputs
│   └── gch.yaml
├── wind_conditions
│   ├── windRose_*.npy
│   └── tiMatrix_*.npy
└── results
```

**Notes:**

- `src/` contains the core implementation.
- `examples/` contains runnable entry points and example parameter files.
- `floris_inputs/` stores FLORIS YAML model inputs.
- `wind_conditions/` stores wind-resource frequency tables and optional turbulence-intensity tables.
- `results/` is used for generated output files.

---

## Installation

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Evaluator modes

### `exact`

Direct FLORIS evaluation on the input wind-rose support.

In this repository, `exact` means direct wake-model evaluation on the supplied WD-WS support of the input wind rose.

### `rank1`

A low-rank approximation based on sparse evaluation of:

- one full WD slice at an energy-selected WS pivot
- one full WS slice at an energy-selected WD pivot

These slices are used to reconstruct the normalized farm-efficiency field.

### `adaptive`

An extension of `rank1` with additional WD basis rows selected using the final adaptive Strategy A.

The basis is built from:

1. global minimum probe direction
2. global maximum representative
3. Strategy-A midpoint
4. pivot direction

If duplicates occur and the number of basis directions is smaller than `adaptive_k`, the remaining slots are filled by descending energy contribution.

### `hybrid`

A staged evaluator:

- early stage: `rank1`
- late stage: `exact`

The switching point is controlled by `hybrid_exact_start_ratio`.

---

## Case types

Two case types are supported.

### `case_type = real`

A realistic wind-resource case loaded from file.

For real cases, the code loads:

- a wind-resource frequency table from `wind_rose_file`
- an optional TI matrix from `ti_file`

Current default discretization in the loader is:

- wind directions: `np.arange(0, 360, 5) - 2.5`
- wind speeds: `np.arange(0.5, 25.0, 1.0)`

### `case_type = classical`

An idealized benchmark case constructed directly in code.

Currently supported classical cases are:

- `case_index = 1`: one wind direction, one wind speed, probability concentrated at a single state
- `case_index = 2`: one wind speed with uniform probability across all wind directions

These cases are useful for debugging, benchmarking, and method validation.

---

## Inputs

The repository expects the following inputs:

1. **FLORIS model input**
   - recommended location: `floris_inputs/gch.yaml`

2. **Wind-resource frequency table**
   - for example: `wind_conditions/windRose_3.npy`

3. **Optional turbulence-intensity table**
   - for example: `wind_conditions/tiMatrix_3.npy`

For real cases, the wind-resource and TI arrays are expected to match the wind-direction and wind-speed discretization used in the optimization setup.

---

## Configuration file

The example configuration file is:

```text
examples/parameters_example.txt
```

It uses a simple format:

```text
key = value
```

Lines starting with `#` or content following `#` are ignored.

---

## Path handling

Relative paths are resolved relative to the configuration-file location.

For example:

```text
floris_input_file = ../floris_inputs/gch.yaml
wind_rose_file = ../wind_conditions/windRose_3.npy
ti_file = ../wind_conditions/tiMatrix_3.npy
results_dir = ../results
```

---

## Main configuration options

### FLORIS and case settings

```text
case_name = real_case3
case_type = real
case_index = 3
floris_input_file = ../floris_inputs/gch.yaml
wind_rose_file = ../wind_conditions/windRose_3.npy
ti_file = ../wind_conditions/tiMatrix_3.npy
```

### Layout settings

```text
Nx = 4
Ny = 4
spacing_D = [6.0, 6.0]
min_spacing_D = 3.0
shape = Rectangle
aspect_ratio = 1.0
```

### Optimization settings

```text
n_individuals = 4
seconds_per_iteration = 60
total_optimization_seconds = 600
n_workers = 1
random_seed = 42
grid_step_size = 50
relegation_number = 1
interface = multiprocessing
```

### Evaluator settings

```text
aep_mode = adaptive
hybrid_exact_start_ratio = 0.5
lowrank_cut_in = 3.0
rated_ws = 12.5
eff_unity_ws_start = 17.0
pivot_ws_min = 4.0
pivot_ws_max = 12.5
adaptive_k = 4
tie_atol_x_max = 1e-6
```

### Wind and TI settings

```text
use_cut_out = 1
cut_out_ws = 25.0
ti_constant = 0.06
```

### Output settings

```text
results_dir = ../results
save_history = 1
```

---

## How wind conditions and TI are loaded

Wind conditions and TI are handled in `src/case_utils.py`.

### Wind resource

Wind-resource loading is controlled by:

- `case_type`
- `case_index`
- `wind_rose_file`

For `case_type = real`, the frequency table is loaded from `wind_rose_file`.

For `case_type = classical`, the frequency table is generated directly in code.

### TI matrix

TI loading is controlled by:

- `ti_file`
- `ti_constant`

If `ti_file` is provided and the file exists, the TI matrix is loaded from file.

If `ti_file` is omitted or set to `None`, the code uses a constant TI field:

```python
ti_table = ti_constant * ones_like(freq_table)
```

So the code supports both:

- case-specific TI matrix loading
- constant-TI fallback without a TI file

---

## Run an example

Run the commands from the repository root.

Basic run:

```bash
python examples/run_single_case.py --config examples/parameters_example.txt
```

Override evaluator mode from the command line:

```bash
python examples/run_single_case.py --config examples/parameters_example.txt --mode exact
python examples/run_single_case.py --config examples/parameters_example.txt --mode rank1
python examples/run_single_case.py --config examples/parameters_example.txt --mode adaptive
python examples/run_single_case.py --config examples/parameters_example.txt --mode hybrid
```

---

## Output

The script saves a result file such as:

```text
results/real_case3_adaptive_result.pkl
```

The saved result dictionary may include:

- `x_opt`: optimized x coordinates
- `y_opt`: optimized y coordinates
- `objective_initial`: initial AEP objective
- `objective_final`: final AEP objective
- `history`: optimization history if enabled
- `history_layout_x_key`: detected x-layout key in history
- `history_layout_y_key`: detected y-layout key in history
- `num_objective_calls_log`: objective-call counts per generation
- `num_accept_log`: accepted-move counts per generation
- `total_objective_calls`: total number of objective evaluations
- `eval_mode_log`: evaluator mode used across generations
- `wind_data_info`: wind-data source and metadata
- `config`: parsed configuration dictionary

The exact content depends on the selected evaluator mode and run settings.

---

## Method notes

The low-rank reconstruction is performed on normalized farm efficiency:

```text
eta(WD, WS) = P_farm(WD, WS) / P_no_wake(WS)
```

The final farm power is then recovered by:

```text
P_farm_approx(WD, WS) = eta_approx(WD, WS) * P_no_wake(WS)
```

Only the final adaptive Strategy A is retained in this repository. Historical adaptive variants are intentionally removed to keep the implementation aligned with the paper-facing version.

---

## Quick sanity check

A simple quick test is to:

1. reduce the optimization budget
2. run a small case under `exact`
3. run the same case under `rank1`
4. confirm that:
   - both runs finish without errors
   - result files are generated
   - final AEP values are finite and positive
   - `hybrid` shows a mode switch in its evaluator log

Suggested temporary settings:

```text
Nx = 3
Ny = 3
n_individuals = 1
seconds_per_iteration = 5
total_optimization_seconds = 10
n_workers = 1
interface = None
```

A minimal benchmark-style test can also use:

```text
case_type = classical
case_index = 1
```

---

## Common issues

### 1. FLORIS import error

Make sure `floris` is installed in the active Python environment.

### 2. Missing FLORIS input file

Check that the FLORIS YAML file exists at the path specified in the configuration file, for example:

```text
floris_input_file = ../floris_inputs/gch.yaml
```

### 3. Missing wind-condition files

Check the file paths in the configuration file.

### 4. TI shape mismatch

The TI table must have the same shape as the frequency table.

### 5. Multiprocessing issues

For debugging, use:

```text
interface = None
n_individuals = 1
n_workers = 1
```

### 6. Real case without TI file

If `ti_file` is not provided, the code automatically falls back to a constant TI field using `ti_constant`.

---

## Suggested requirements.txt

A minimal requirements file should include at least:

```text
numpy
scipy
shapely
floris
```

You may pin versions depending on your local environment and FLORIS compatibility.

---

## Citation

If you use this repository in academic work, please cite the corresponding paper once available.

---

## License

Add your preferred open-source license here before public release.

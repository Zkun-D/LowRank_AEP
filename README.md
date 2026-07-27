# LowRank_AEP

Reference implementation of the sparse **Rank-1** and **Adaptive** evaluators
used for accelerated wind-farm annual energy production (AEP) evaluation with
FLORIS.

The associated manuscript is titled:

> **Accelerating Wind-Farm AEP Evaluation by Exploiting Near-Low-Rank Wake Structure**

## Repository scope

This release contains the core evaluators, runnable FLORIS-based examples, and
an illustrative Rank-1-to-Exact hybrid random-search driver. It does **not**
include the full HPC orchestration, multi-start SLSQP batches, statistical
analysis, or manuscript figure-generation workflow.

## Evaluators

- `exact`: direct FLORIS evaluation on the supplied wind-resource support.
- `rank1`: sparse cross sampling and Rank-1 reconstruction of normalized farm
  efficiency.
- `adaptive`: nested, pivot-preserving Strategy-A basis refinement.
- `hybrid`: Rank-1 during the early search and Exact during the late search.

The Adaptive hierarchy is:

- `K=1`: pivot only, exactly reproducing Rank-1;
- `K=2`: pivot plus the farther probe-space extreme;
- `K=3`: pivot plus both extremes;
- `K=4`: pivot, both extremes, and the intermediate representative;
- `K>4`: additional directions selected by maximin fill by default.

Adaptive Stage II reuses states already evaluated by the Stage-I cross. Final
layouts produced by the example optimizer are also re-evaluated with Exact
FLORIS before saving.

## Installation

The reference environment used Python 3.9 with FLORIS 4.2.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run an example

From the repository root:

```bash
python examples/run_single_case.py --config examples/parameters_example.txt --mode rank1
python examples/run_single_case.py --config examples/parameters_example.txt --mode adaptive
python examples/run_single_case.py --config examples/parameters_example.txt --mode exact
python examples/run_single_case.py --config examples/parameters_example.txt --mode hybrid
```

The example configuration uses the included wind-resource and turbulence-
intensity files. Relative paths are resolved from the configuration file.

A very small four-mode smoke test is available through:

```bash
python examples/test_run_modes.py
```

## Important configuration options

```text
aep_mode = adaptive
lowrank_cut_in = 3.0
eff_unity_ws_start = 17.0
pivot_ws_min = 4.0
pivot_ws_max = 12.5
adaptive_k = 4
adaptive_fill_strategy = maximin
reuse_stage1_overlap = 1
```

## Saved result values

The example driver saves both:

- `objective_final_search`: terminal objective under the search evaluator;
- `objective_final_exact`: the same terminal layout re-evaluated with Exact
  FLORIS.

The legacy key `objective_final` is retained as an alias for the search value.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The included tests verify the nested, pivot-preserving Adaptive hierarchy,
configuration paths, and bundled example arrays.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated
paper when it becomes available.

## License

MIT License. Copyright (c) 2026 DONG Zhikun and DENG Xiaowei.

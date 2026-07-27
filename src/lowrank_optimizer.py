"""
Low-rank random-search wind-farm layout optimization.

Implemented evaluator modes:
- exact
- rank1
- adaptive
- hybrid

Notes
-----
1. "Efficiency" refers to normalized farm power:
       eta(wd, ws) = P_farm(wd, ws) / P_no_wake(ws)

2. Rank1 and adaptive evaluators reconstruct eta in sparse-query form and then
   recover farm power by multiplying with P_no_wake(ws).

3. The hybrid mode uses rank1 in the early stage of optimization and exact
   evaluation in the late stage.
"""

from __future__ import annotations

import numpy as np
from time import perf_counter as timerpc
from scipy.spatial.distance import cdist, pdist
from shapely.geometry import Point
from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_base import (
    LayoutOptimization,
)
from adaptive_basis import select_adaptive_basis_strategy_A


def _load_local_floris_object(fmodel_dict, wind_data=None):
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(wind_data=wind_data)
    return fmodel


def test_min_dist(layout_x, layout_y, min_dist):
    coords = np.array([layout_x, layout_y]).T
    dist = pdist(coords)
    return dist.min() >= min_dist


def test_point_in_bounds(test_x, test_y, poly_outer):
    return poly_outer.contains(Point(test_x, test_y))


def _gen_dist_based_init(N, step_size, poly_outer, min_x, max_x, min_y, max_y, seed):
    if seed is not None:
        np.random.seed(seed)

    init_x = float(np.random.randint(int(min_x), int(max_x)))
    init_y = float(np.random.randint(int(min_y), int(max_y)))

    while not poly_outer.contains(Point([init_x, init_y])):
        init_x = float(np.random.randint(int(min_x), int(max_x)))
        init_y = float(np.random.randint(int(min_y), int(max_y)))

    layout_x = np.array([init_x])
    layout_y = np.array([init_y])
    layout = np.array([layout_x, layout_y]).T

    for _ in range(1, N):
        max_dist = 0.0
        save_x = None
        save_y = None

        for x in np.arange(min_x, max_x, step_size):
            for y in np.arange(min_y, max_y, step_size):
                if poly_outer.contains(Point([x, y])):
                    test_dist = cdist([[x, y]], layout)
                    min_dist_val = np.min(test_dist)
                    if min_dist_val > max_dist:
                        max_dist = min_dist_val
                        save_x = x
                        save_y = y

        layout_x = np.append(layout_x, [save_x])
        layout_y = np.append(layout_y, [save_y])
        layout = np.array([layout_x, layout_y]).T

    return layout_x, layout_y


# Adaptive basis selection is implemented in adaptive_basis.py.

# ============================================================
# Evaluator
# ============================================================

def _return_evaluator_result(aep_value, diagnostics, return_diagnostics):
    aep_value = float(aep_value)
    return (aep_value, diagnostics) if return_diagnostics else aep_value


def compute_aep_with_evaluator(
    layout_x,
    layout_y,
    fmodel_dict,
    wind_directions,
    wind_speeds,
    ti_table,
    freq_table,
    p_no_wake_1d,
    lowrank_cut_in,
    eff_unity_ws_start,
    mode,
    precomputed_pivots,
    effective_n_ws,
    force_low_speed_correction=False,
    correction_threshold_ws=5.0,
    low_speed_anchor=3.5,
    adaptive_k=4,
    tie_atol_x_max=1e-6,
    adaptive_fill_strategy="maximin",
    reuse_stage1_overlap=True,
    return_diagnostics=False,
):
    """Compute AEP using Exact, Rank-1, or Adaptive evaluation.

    ``reuse_stage1_overlap=True`` prevents Adaptive Stage II from recomputing
    states already evaluated in the Stage-I cross.  The default nested basis
    follows the paper-facing V4.2 definition.
    """
    n_wd = len(wind_directions)
    n_ws = len(wind_speeds)
    ws_array = np.asarray(wind_speeds, dtype=float)
    freq_table = np.asarray(freq_table, dtype=float)

    marginal_ws_prob = np.sum(freq_table, axis=0)
    total_prob = np.sum(marginal_ws_prob)
    site_mean_ws = (
        np.sum(ws_array * marginal_ws_prob) / total_prob
        if total_prob > 0 else 0.0
    )
    use_anchor = bool(
        force_low_speed_correction and site_mean_ws < correction_threshold_ws
    )
    idx_anchor = (
        int(np.argmin(np.abs(ws_array - low_speed_anchor))) if use_anchor else -1
    )

    valid_ws_idx = np.where(
        (ws_array >= lowrank_cut_in) & (ws_array <= eff_unity_ws_start)
    )[0]
    valid_ws_idx = valid_ws_idx[valid_ws_idx < effective_n_ws]

    diagnostics = {
        "mode": str(mode),
        "site_mean_ws": float(site_mean_ws),
        "low_speed_correction": bool(use_anchor),
        "basis_indices": np.array([], dtype=int),
        "raw_query_count": 0,
        "unique_query_count": 0,
        "reused_query_count": 0,
    }

    if len(valid_ws_idx) == 0 and mode != "exact":
        return _return_evaluator_result(0.0, diagnostics, return_diagnostics)

    p_no_wake_1d = np.asarray(p_no_wake_1d, dtype=float)
    p_no_wake_2d = np.tile(p_no_wake_1d, (n_wd, 1))

    if mode == "exact":
        wind_rose_full = WindRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            ti_table=ti_table,
            freq_table=freq_table,
        )
        fmodel_full = FlorisModel(fmodel_dict)
        fmodel_full.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose_full)
        fmodel_full.run()
        power_full = fmodel_full.get_farm_power()
        support = int(np.sum(freq_table > 0))
        diagnostics.update(
            raw_query_count=support,
            unique_query_count=support,
        )
        aep_value = np.nansum(power_full * freq_table) * 365.0 * 24.0
        return _return_evaluator_result(aep_value, diagnostics, return_diagnostics)

    if mode not in {"rank1", "adaptive"}:
        raise ValueError(f"Unsupported evaluator mode: {mode}")

    # Stage I: sparse cross.
    u_opt_idx = int(precomputed_pivots["u_opt_idx"])
    v_opt_idx = int(precomputed_pivots["v_opt_idx"])
    diagnostics["pivot_ws_index"] = u_opt_idx
    diagnostics["pivot_wd_index"] = v_opt_idx

    cross_mask = np.zeros((n_wd, n_ws), dtype=bool)
    cross_mask[:, u_opt_idx] = True
    cross_mask[v_opt_idx, :effective_n_ws] = True
    if use_anchor:
        cross_mask[:, idx_anchor] = True

    wind_rose_1 = WindRose(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        ti_table=ti_table,
        freq_table=cross_mask.astype(float),
    )
    fmodel_temp = FlorisModel(fmodel_dict)
    fmodel_temp.set(layout_x=layout_x, layout_y=layout_y, wind_data=wind_rose_1)
    fmodel_temp.run()
    p_res_1 = np.asarray(fmodel_temp.get_farm_power(), dtype=float)

    power_row_u0 = p_res_1[:, u_opt_idx].flatten()
    power_col_v0 = p_res_1[v_opt_idx, :].flatten()
    if effective_n_ws < n_ws:
        power_col_v0[effective_n_ws:] = power_col_v0[effective_n_ws - 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_col = np.divide(
            power_col_v0,
            p_no_wake_1d,
            out=np.zeros_like(power_col_v0),
            where=p_no_wake_1d > 1e-6,
        )
        norm_row = power_row_u0 / p_no_wake_1d[u_opt_idx]

    pivot_val = float(norm_col[u_opt_idx])
    if pivot_val < 1e-8:
        pivot_val = 1.0

    eta_rank1 = np.ones((n_wd, n_ws), dtype=float)
    eta_rank1[:, valid_ws_idx] = (
        np.outer(norm_row, norm_col[valid_ws_idx]) / pivot_val
    )
    eta_rank1[:, ws_array > eff_unity_ws_start] = 1.0
    eta_rank1[:, ws_array < lowrank_cut_in] = 1.0

    if use_anchor:
        true_eff_anchor = np.divide(
            p_res_1[:, idx_anchor],
            p_no_wake_1d[idx_anchor],
            out=np.zeros(n_wd),
            where=p_no_wake_1d[idx_anchor] > 1e-6,
        )
        eta_rank1[:, idx_anchor] = true_eff_anchor
    eta_rank1 = np.clip(eta_rank1, 0.0, 1.0)

    cross_calls = int(np.sum(cross_mask))
    diagnostics.update(
        raw_query_count=cross_calls,
        unique_query_count=cross_calls,
    )

    if mode == "rank1":
        aep_value = np.nansum(eta_rank1 * p_no_wake_2d * freq_table) * 8760.0
        return _return_evaluator_result(aep_value, diagnostics, return_diagnostics)

    energy_per_wd = np.sum(p_no_wake_2d * freq_table, axis=1)
    probe = np.asarray(norm_row, dtype=float)
    basis_indices, idx_min, idx_mid, idx_max = select_adaptive_basis_strategy_A(
        probe=probe,
        energy_per_wd=energy_per_wd,
        pivot_idx=v_opt_idx,
        wd_array=wind_directions,
        k=int(adaptive_k),
        atol_x_max=float(tie_atol_x_max),
        fill_strategy=adaptive_fill_strategy,
    )
    diagnostics.update(
        basis_indices=basis_indices.copy(),
        basis_min_index=int(idx_min),
        basis_mid_index=int(idx_mid),
        basis_max_index=int(idx_max),
    )

    # By definition, Adaptive K=1 is exactly Rank-1.
    if int(adaptive_k) == 1:
        aep_value = np.nansum(eta_rank1 * p_no_wake_2d * freq_table) * 8760.0
        return _return_evaluator_result(aep_value, diagnostics, return_diagnostics)

    requested_basis_mask = np.zeros((n_wd, n_ws), dtype=bool)
    requested_basis_mask[basis_indices, :effective_n_ws] = True
    stage2_mask = (
        requested_basis_mask & ~cross_mask
        if reuse_stage1_overlap
        else requested_basis_mask
    )

    p_res_2 = np.zeros((n_wd, n_ws), dtype=float)
    if np.any(stage2_mask):
        wind_rose_2 = WindRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            ti_table=ti_table,
            freq_table=stage2_mask.astype(float),
        )
        fmodel_temp.set(wind_data=wind_rose_2)
        fmodel_temp.run()
        p_res_2 = np.asarray(fmodel_temp.get_farm_power(), dtype=float)

    power_basis = np.zeros((len(basis_indices), n_ws), dtype=float)
    for row_pos, wd_idx in enumerate(basis_indices):
        wd_idx = int(wd_idx)
        known = cross_mask[wd_idx]
        new = stage2_mask[wd_idx]
        power_basis[row_pos, known] = p_res_1[wd_idx, known]
        power_basis[row_pos, new] = p_res_2[wd_idx, new]

    if effective_n_ws < n_ws:
        power_basis[:, effective_n_ws:] = power_basis[:, effective_n_ws - 1][:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        basis_eta = np.divide(
            power_basis,
            p_no_wake_1d,
            out=np.zeros_like(power_basis),
            where=p_no_wake_1d > 1e-6,
        )

    x_nodes = probe[basis_indices]
    sort_idx = np.argsort(x_nodes)
    x_sorted = x_nodes[sort_idx]
    x_unique, uniq_idx = np.unique(x_sorted, return_index=True)

    eta_adaptive = np.ones_like(eta_rank1)
    eta_adaptive[:, ws_array > eff_unity_ws_start] = 1.0
    eta_adaptive[:, ws_array < lowrank_cut_in] = 1.0
    if x_unique.size == 1:
        for j in valid_ws_idx:
            y_sorted = basis_eta[:, j][sort_idx]
            eta_adaptive[:, j] = float(y_sorted[int(uniq_idx[0])])
    else:
        for j in valid_ws_idx:
            y_sorted = basis_eta[:, j][sort_idx][uniq_idx]
            eta_adaptive[:, j] = np.interp(
                probe,
                x_unique,
                y_sorted,
                left=float(y_sorted[0]),
                right=float(y_sorted[-1]),
            )

    if use_anchor:
        true_eff_anchor = np.divide(
            p_res_1[:, idx_anchor],
            p_no_wake_1d[idx_anchor],
            out=np.zeros(n_wd),
            where=p_no_wake_1d[idx_anchor] > 1e-6,
        )
        eta_adaptive[:, idx_anchor] = true_eff_anchor
    eta_adaptive = np.clip(eta_adaptive, 0.0, 1.0)

    unique_calls = int(np.sum(cross_mask | requested_basis_mask))
    raw_calls = cross_calls + int(np.sum(stage2_mask))
    diagnostics.update(
        raw_query_count=raw_calls,
        unique_query_count=unique_calls,
        reused_query_count=int(np.sum(cross_mask & requested_basis_mask))
        if reuse_stage1_overlap else 0,
    )

    aep_value = np.nansum(eta_adaptive * p_no_wake_2d * freq_table) * 8760.0
    return _return_evaluator_result(aep_value, diagnostics, return_diagnostics)

def run_single_individual_search(
    seconds_per_iteration,
    initial_objective,
    layout_x,
    layout_y,
    fmodel_dict,
    wind_directions,
    wind_speeds,
    ti_table,
    freq_table,
    min_dist,
    poly_outer,
    dist_pmf,
    enable_geometric_yaw,
    seed,
    p_no_wake_1d,
    lowrank_cut_in,
    eff_unity_ws_start,
    mode,
    precomputed_pivots,
    effective_n_ws,
    adaptive_k,
    tie_atol_x_max,
    adaptive_fill_strategy,
    reuse_stage1_overlap,
):
    if enable_geometric_yaw:
        raise NotImplementedError(
            "Geometric-yaw optimization is not part of this public release."
        )

    if seed is not None:
        np.random.seed(seed)

    single_opt_start_time = timerpc()
    stop_time = single_opt_start_time + seconds_per_iteration

    num_objective_calls = 0
    num_accept = 0
    num_turbines = len(layout_x)
    current_objective = initial_objective

    use_momentum = False
    get_new_point = True

    while timerpc() < stop_time:
        if not use_momentum:
            get_new_point = True

        if get_new_point:
            tr = np.random.randint(0, num_turbines)
            rand_dir = np.random.uniform(0.0, 2.0 * np.pi)
            rand_dist = np.random.choice(dist_pmf["d"], p=dist_pmf["p"])

            test_x = layout_x[tr] + np.cos(rand_dir) * rand_dist
            test_y = layout_y[tr] + np.sin(rand_dir) * rand_dist

            if not test_point_in_bounds(test_x, test_y, poly_outer):
                continue

            original_x, original_y = layout_x[tr], layout_y[tr]
            layout_x[tr] = test_x
            layout_y[tr] = test_y

            if not test_min_dist(layout_x, layout_y, min_dist):
                layout_x[tr] = original_x
                layout_y[tr] = original_y
                continue

            num_objective_calls += 1
            test_objective = compute_aep_with_evaluator(
                layout_x=layout_x,
                layout_y=layout_y,
                fmodel_dict=fmodel_dict,
                wind_directions=wind_directions,
                wind_speeds=wind_speeds,
                ti_table=ti_table,
                freq_table=freq_table,
                p_no_wake_1d=p_no_wake_1d,
                lowrank_cut_in=lowrank_cut_in,
                eff_unity_ws_start=eff_unity_ws_start,
                mode=mode,
                precomputed_pivots=precomputed_pivots,
                effective_n_ws=effective_n_ws,
                force_low_speed_correction=True,
                adaptive_k=int(adaptive_k),
                tie_atol_x_max=float(tie_atol_x_max),
                adaptive_fill_strategy=adaptive_fill_strategy,
                reuse_stage1_overlap=bool(reuse_stage1_overlap),
            )

        if test_objective > current_objective:
            current_objective = test_objective
            num_accept += 1
            get_new_point = False
        else:
            layout_x[tr] = original_x
            layout_y[tr] = original_y

    return current_objective, layout_x, layout_y, num_objective_calls, num_accept


class LayoutOptimizationRandomSearchLowRank(LayoutOptimization):
    def __init__(
        self,
        fmodel,
        boundaries,
        min_dist=None,
        min_dist_D=None,
        distance_pmf=None,
        n_individuals=4,
        seconds_per_iteration=60.0,
        total_optimization_seconds=600.0,
        interface="multiprocessing",
        max_workers=None,
        grid_step_size=100.0,
        relegation_number=1,
        enable_geometric_yaw=False,
        use_dist_based_init=True,
        random_seed=None,
        aep_mode="rank1",
        hybrid_exact_start_ratio=0.5,
        lowrank_cut_in=3.0,
        rated_ws=12.5,
        eff_unity_ws_start=17.0,
        pivot_ws_min=4.0,
        pivot_ws_max=12.5,
        adaptive_k=4,
        tie_atol_x_max=1e-6,
        adaptive_fill_strategy="maximin",
        reuse_stage1_overlap=True,
    ):
        valid_modes = ["exact", "rank1", "adaptive", "hybrid"]
        if aep_mode not in valid_modes:
            raise ValueError(f"aep_mode must be one of {valid_modes}")

        self.aep_mode = aep_mode
        self.lowrank_cut_in = float(lowrank_cut_in)
        self.rated_ws = float(rated_ws)
        self.eff_unity_ws_start = float(eff_unity_ws_start)
        self.pivot_ws_min = float(pivot_ws_min)
        self.pivot_ws_max = float(pivot_ws_max)
        self.adaptive_k = int(adaptive_k)
        self.tie_atol_x_max = float(tie_atol_x_max)
        self.adaptive_fill_strategy = str(adaptive_fill_strategy).lower()
        self.reuse_stage1_overlap = bool(reuse_stage1_overlap)
        if self.adaptive_fill_strategy not in {"maximin", "energy"}:
            raise ValueError("adaptive_fill_strategy must be 'maximin' or 'energy'.")
        if enable_geometric_yaw:
            raise NotImplementedError(
                "Geometric-yaw optimization is not part of this public release."
            )
        self.hybrid_exact_start_ratio = float(hybrid_exact_start_ratio)

        if not (0.0 <= self.hybrid_exact_start_ratio <= 1.0):
            raise ValueError("hybrid_exact_start_ratio must be in [0, 1].")

        if isinstance(interface, str) and interface.lower() == "none":
            interface = None

        if interface == "mpi4py":
            import mpi4py.futures as mp
            self._PoolExecutor = mp.MPIPoolExecutor
        elif interface == "multiprocessing":
            import multiprocessing as mp
            self._PoolExecutor = mp.Pool
            if max_workers is None:
                max_workers = mp.cpu_count()
        elif interface is None:
            if n_individuals > 1 or (max_workers is not None and max_workers > 1):
                print("Parallelization not available with interface=None. Reducing n_individuals to 1.")
                n_individuals = 1
            self._PoolExecutor = None
            max_workers = None
        else:
            raise ValueError(f"Interface '{interface}' not recognized.")

        self.max_workers = max_workers
        self.interface = interface
        self.random_seed = random_seed
        self.history = []

        if relegation_number > n_individuals / 2:
            raise ValueError("relegation_number must be less than n_individuals / 2.")
        self.relegation_number = relegation_number

        self.D = fmodel.core.farm.rotor_diameters.max()
        self.N_turbines = fmodel.n_turbines

        if min_dist is not None and min_dist_D is not None:
            raise ValueError("Only one of min_dist and min_dist_D can be defined.")
        if min_dist_D is not None:
            min_dist = min_dist_D * self.D

        super().__init__(
            fmodel,
            boundaries,
            min_dist=min_dist,
            enable_geometric_yaw=enable_geometric_yaw,
            use_value=False,
        )

        self._obj_name = "AEP"
        self._obj_unit = "[GWh]"
        self.min_dist_D = self.min_dist / self.D
        self._process_dist_pmf(distance_pmf)
        self.fmodel_dict = self.fmodel.core.as_dict()
        self.grid_step_size = grid_step_size
        self.n_individuals = n_individuals
        self.x_initial = self.fmodel.layout_x
        self.y_initial = self.fmodel.layout_y
        self.total_optimization_seconds = total_optimization_seconds
        self.seconds_per_iteration = seconds_per_iteration

        self.wind_directions = np.array(self.fmodel.wind_data.wind_directions)
        self.wind_speeds = np.array(self.fmodel.wind_data.wind_speeds)
        self.ti_table = np.array(self.fmodel.wind_data.ti_table)
        self.freq_table = np.array(self.fmodel.wind_data.freq_table)

        marginal_prob = np.sum(self.freq_table, axis=0)
        valid_ws_indices = np.where(marginal_prob > 0)[0]
        if len(valid_ws_indices) > 0:
            self.effective_n_ws = valid_ws_indices[-1] + 1
            self.effective_n_ws = min(len(self.wind_speeds), self.effective_n_ws + 1)
        else:
            self.effective_n_ws = len(self.wind_speeds)

        print(f"Effective wind-speed bins: {self.effective_n_ws} / {len(self.wind_speeds)}")

        self.p_no_wake_1d = self._precompute_no_wake_power_vector(rated_ws=self.rated_ws)
        self.precomputed_pivots = self._precompute_energy_weighted_pivots(
            pivot_ws_min=self.pivot_ws_min,
            pivot_ws_max=self.pivot_ws_max,
            use_symmetric_pivot=True,
        )

        print("[Evaluator Params]")
        print(f"  mode                    = {self.aep_mode}")
        print(f"  lowrank_cut_in          = {self.lowrank_cut_in:.2f} m/s")
        print(f"  rated_ws                = {self.rated_ws:.2f} m/s")
        print(f"  eff_unity_ws_start      = {self.eff_unity_ws_start:.2f} m/s")
        print(f"  pivot_ws_min/max        = {self.pivot_ws_min:.2f} / {self.pivot_ws_max:.2f} m/s")
        print(f"  u_opt_idx / v_opt_idx   = {self.precomputed_pivots['u_opt_idx']} / {self.precomputed_pivots['v_opt_idx']}")
        if self.aep_mode == "hybrid":
            print(f"  hybrid_exact_start_ratio= {self.hybrid_exact_start_ratio:.3f}")

        init_mode = self._get_init_evaluator_mode()
        self.objective_initial = compute_aep_with_evaluator(
            layout_x=self.x_initial,
            layout_y=self.y_initial,
            fmodel_dict=self.fmodel_dict,
            wind_directions=self.wind_directions,
            wind_speeds=self.wind_speeds,
            ti_table=self.ti_table,
            freq_table=self.freq_table,
            p_no_wake_1d=self.p_no_wake_1d,
            lowrank_cut_in=self.lowrank_cut_in,
            eff_unity_ws_start=self.eff_unity_ws_start,
            mode=init_mode,
            precomputed_pivots=self.precomputed_pivots,
            effective_n_ws=self.effective_n_ws,
            force_low_speed_correction=True,
            adaptive_k=self.adaptive_k,
            tie_atol_x_max=self.tie_atol_x_max,
            adaptive_fill_strategy=self.adaptive_fill_strategy,
            reuse_stage1_overlap=self.reuse_stage1_overlap,
        )

        self.objective_mean = self.objective_initial
        self.objective_median = self.objective_initial
        self.objective_max = self.objective_initial
        self.objective_min = self.objective_initial

        self.x_candidate = np.zeros((self.n_individuals, self.N_turbines))
        self.y_candidate = np.zeros((self.n_individuals, self.N_turbines))
        self.objective_candidate = np.zeros(self.n_individuals)
        self.iteration_step = -1
        self.opt_time_start = timerpc()
        self.opt_time = 0.0

        if use_dist_based_init:
            self._generate_initial_layouts()
        else:
            for i in range(self.n_individuals):
                self.x_candidate[i, :] = self.x_initial
                self.y_candidate[i, :] = self.y_initial
                self.objective_candidate[i] = self.objective_initial

        self._evaluate_opt_step()

    def _get_init_evaluator_mode(self):
        if self.aep_mode == "hybrid":
            return "rank1"
        return self.aep_mode

    def _get_current_evaluator_mode(self):
        if self.aep_mode != "hybrid":
            return self.aep_mode

        elapsed = timerpc() - self._opt_start_time
        if self.total_optimization_seconds <= 0:
            return "exact"

        ratio = elapsed / self.total_optimization_seconds
        return "rank1" if ratio < self.hybrid_exact_start_ratio else "exact"

    def _precompute_no_wake_power_vector(self, rated_ws):
        wd = np.array(self.fmodel.wind_data.wind_directions)
        ws = np.array(self.fmodel.wind_data.wind_speeds)
        ti_table = np.array(self.fmodel.wind_data.ti_table)

        selected_wd = np.array([wd[0]])
        selected_ti = ti_table[0:1, :]

        wind_rose_no_wake = WindRose(
            wind_directions=selected_wd,
            wind_speeds=ws,
            ti_table=selected_ti,
            freq_table=np.ones((1, len(ws))) / len(ws),
        )

        fmodel_no_wake = FlorisModel(self.fmodel_dict)
        fmodel_no_wake.set(
            layout_x=self.x_initial,
            layout_y=self.y_initial,
            wind_data=wind_rose_no_wake,
        )
        fmodel_no_wake.run_no_wake()
        power_no_wake_1d = fmodel_no_wake.get_farm_power().flatten()

        rated_mask = ws >= rated_ws
        if np.any(rated_mask):
            rated_power_val = power_no_wake_1d[rated_mask][0]
            power_no_wake_1d[rated_mask] = rated_power_val

        return power_no_wake_1d

    def _precompute_energy_weighted_pivots(
        self,
        pivot_ws_min=5.0,
        pivot_ws_max=12.0,
        use_symmetric_pivot=True,
    ):
        ws_array = np.array(self.wind_speeds)

        p_no_wake_2d = np.tile(self.p_no_wake_1d, (len(self.wind_directions), 1))
        e_weight = p_no_wake_2d * self.freq_table

        valid_mask = (ws_array >= max(self.lowrank_cut_in, pivot_ws_min)) & (ws_array <= pivot_ws_max)
        energy_per_ws = np.sum(e_weight, axis=0)
        energy_per_ws_masked = energy_per_ws.copy()
        energy_per_ws_masked[~valid_mask] = -1.0

        if np.all(energy_per_ws_masked < 0):
            valid_mask_fb = (ws_array >= self.lowrank_cut_in) & (ws_array <= self.eff_unity_ws_start)
            energy_per_ws_masked = energy_per_ws.copy()
            energy_per_ws_masked[~valid_mask_fb] = -1.0

        u_opt_idx = int(np.argmax(energy_per_ws_masked))

        energy_per_wd = np.sum(e_weight, axis=1)
        if use_symmetric_pivot:
            n_wd = len(self.wind_directions)
            offset = n_wd // 2
            symmetric_score = energy_per_wd + np.roll(energy_per_wd, -offset)
            v_opt_idx = int(np.argmax(symmetric_score))
        else:
            v_opt_idx = int(np.argmax(energy_per_wd))

        return {
            "u_opt_idx": u_opt_idx,
            "v_opt_idx": v_opt_idx,
            "pivot_ws_min": float(pivot_ws_min),
            "pivot_ws_max": float(pivot_ws_max),
        }

    def _process_dist_pmf(self, dist_pmf):
        if dist_pmf is None:
            jump_dist = np.min([self.xmax - self.xmin, self.ymax - self.ymin]) / 2.0
            jump_prob = 0.05
            d = np.append(np.linspace(0.0, 2.0 * self.D, 99), jump_dist)
            p = np.append((1 - jump_prob) / len(d) * np.ones(len(d) - 1), jump_prob)
            p = p / p.sum()
            dist_pmf = {"d": d, "p": p}

        if not all(k in dist_pmf for k in ("d", "p")):
            raise KeyError('distance_pmf must contain keys "d" and "p".')

        if not hasattr(dist_pmf["d"], "__len__") or len(dist_pmf["d"]) != len(dist_pmf["p"]):
            raise TypeError("distance_pmf entries should be arrays/lists of equal length.")

        if not np.isclose(np.sum(dist_pmf["p"]), 1.0):
            dist_pmf["p"] = np.array(dist_pmf["p"]) / np.sum(dist_pmf["p"])

        self.distance_pmf = dist_pmf

    def _generate_initial_layouts(self):
        if self.random_seed is None:
            seeds = [None] * self.n_individuals
        else:
            base_seed = int(self.random_seed) + 23
            seeds = [base_seed + i for i in range(self.n_individuals)]

        print(f"Generating {self.n_individuals} initial layouts...")
        t1 = timerpc()

        multiargs = [
            (
                self.N_turbines,
                self.grid_step_size,
                self._boundary_polygon,
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
                seeds[i],
            )
            for i in range(self.n_individuals)
        ]

        if self._PoolExecutor:
            with self._PoolExecutor(self.max_workers) as p:
                out = p.starmap(_gen_dist_based_init, multiargs)
        else:
            out = [_gen_dist_based_init(*args) for args in multiargs]

        init_mode = self._get_init_evaluator_mode()

        for i in range(self.n_individuals):
            self.x_candidate[i, :] = out[i][0]
            self.y_candidate[i, :] = out[i][1]

            self.objective_candidate[i] = compute_aep_with_evaluator(
                layout_x=self.x_candidate[i, :],
                layout_y=self.y_candidate[i, :],
                fmodel_dict=self.fmodel_dict,
                wind_directions=self.wind_directions,
                wind_speeds=self.wind_speeds,
                ti_table=self.ti_table,
                freq_table=self.freq_table,
                p_no_wake_1d=self.p_no_wake_1d,
                lowrank_cut_in=self.lowrank_cut_in,
                eff_unity_ws_start=self.eff_unity_ws_start,
                mode=init_mode,
                precomputed_pivots=self.precomputed_pivots,
                effective_n_ws=self.effective_n_ws,
                force_low_speed_correction=True,
                adaptive_k=self.adaptive_k,
                tie_atol_x_max=self.tie_atol_x_max,
                adaptive_fill_strategy=self.adaptive_fill_strategy,
                reuse_stage1_overlap=self.reuse_stage1_overlap,
            )

        t2 = timerpc()
        print(f"Time to generate initial layouts: {t2 - t1:.3f} s")

    def _evaluate_opt_step(self):
        sorted_indices = np.argsort(self.objective_candidate)[::-1]
        self.objective_candidate = self.objective_candidate[sorted_indices]
        self.x_candidate = self.x_candidate[sorted_indices]
        self.y_candidate = self.y_candidate[sorted_indices]

        self.opt_time = timerpc() - self.opt_time_start
        self.iteration_step += 1

        self.objective_mean = np.mean(self.objective_candidate)
        self.objective_median = np.median(self.objective_candidate)
        self.objective_max = np.max(self.objective_candidate)
        self.objective_min = np.min(self.objective_candidate)

        increase_mean = 100.0 * (self.objective_mean - self.objective_initial) / self.objective_initial
        increase_max = 100.0 * (self.objective_max - self.objective_initial) / self.objective_initial

        print("=======================================")
        print(f"Step {self.iteration_step:+.1f} | Time {self.opt_time:+.1f}s")
        print(f"Mean {self._obj_name} = {self.objective_mean / 1e9:.6f} ({increase_mean:+.4f}%)")
        print(f"Max  {self._obj_name} = {self.objective_max / 1e9:.6f} ({increase_max:+.4f}%)")
        if hasattr(self, "_num_accept") and self._num_accept is not None:
            total_accept = int(np.sum(self._num_accept))
            mean_accept = float(np.mean(self._num_accept))
            print(f"Accept moves: total={total_accept:d}, mean/ind={mean_accept:.2f}")
        print("=======================================")

        if self.relegation_number > 0:
            self.objective_candidate[-self.relegation_number:] = self.objective_candidate[:self.relegation_number]
            self.x_candidate[-self.relegation_number:] = self.x_candidate[:self.relegation_number]
            self.y_candidate[-self.relegation_number:] = self.y_candidate[:self.relegation_number]

        if hasattr(self, "_num_accept") and self._num_accept is not None:
            accept_per_ind = np.array(self._num_accept, dtype=int).copy()
            accept_total = int(np.sum(accept_per_ind))
            accept_mean = float(np.mean(accept_per_ind))
        else:
            accept_per_ind = None
            accept_total = None
            accept_mean = None

        self.history.append({
            "iteration": self.iteration_step,
            "time": self.opt_time,
            "objective_mean": float(self.objective_mean),
            "objective_max": float(self.objective_max),
            "layouts_x": self.x_candidate.copy(),
            "layouts_y": self.y_candidate.copy(),
            "eval_mode": getattr(self, "_current_generation_mode", self.aep_mode),
            "accept_total": accept_total,
            "accept_mean_per_individual": accept_mean,
            "accept_per_individual": accept_per_ind,
        })

    def _initialize_optimization(self):
        print(f"Optimizing using {self.n_individuals} individuals.")
        self._opt_start_time = timerpc()
        self.opt_time_start = self._opt_start_time
        self._opt_stop_time = self._opt_start_time + self.total_optimization_seconds

        self.objective_candidate_log = [self.objective_candidate.copy()]
        self.num_objective_calls_log = []
        self.num_accept_log = []
        self.total_objective_calls = 0
        self.eval_mode_log = []

        self._num_objective_calls = [0] * self.n_individuals
        self._num_accept = [0] * self.n_individuals
        self._current_generation_mode = self._get_init_evaluator_mode()
        self._population_mode = self._get_init_evaluator_mode()

    def _rescore_population(self, mode):
        print(f"[RESCORE] Re-evaluating population using mode='{mode}' ...")
        for i in range(self.n_individuals):
            self.objective_candidate[i] = compute_aep_with_evaluator(
                layout_x=self.x_candidate[i, :],
                layout_y=self.y_candidate[i, :],
                fmodel_dict=self.fmodel_dict,
                wind_directions=self.wind_directions,
                wind_speeds=self.wind_speeds,
                ti_table=self.ti_table,
                freq_table=self.freq_table,
                p_no_wake_1d=self.p_no_wake_1d,
                lowrank_cut_in=self.lowrank_cut_in,
                eff_unity_ws_start=self.eff_unity_ws_start,
                mode=mode,
                precomputed_pivots=self.precomputed_pivots,
                effective_n_ws=self.effective_n_ws,
                force_low_speed_correction=True,
                adaptive_k=self.adaptive_k,
                tie_atol_x_max=self.tie_atol_x_max,
                adaptive_fill_strategy=self.adaptive_fill_strategy,
                reuse_stage1_overlap=self.reuse_stage1_overlap,
            )

    def _sort_population_by_objective(self):
        sorted_indices = np.argsort(self.objective_candidate)[::-1]
        self.objective_candidate = self.objective_candidate[sorted_indices]
        self.x_candidate = self.x_candidate[sorted_indices]
        self.y_candidate = self.y_candidate[sorted_indices]

    def _append_history_event_mode_switch(self, prev_mode, new_mode, objective_before):
        event = {
            "iteration": self.iteration_step,
            "time": float(timerpc() - self.opt_time_start),
            "event": "mode_switch_rescore",
            "prev_mode": str(prev_mode),
            "new_mode": str(new_mode),
            "objective_mean_before": float(np.mean(objective_before)),
            "objective_max_before": float(np.max(objective_before)),
            "objective_mean_after": float(np.mean(self.objective_candidate)),
            "objective_max_after": float(np.max(self.objective_candidate)),
            "objective_candidates_before": np.array(objective_before, float).copy(),
            "objective_candidates_after": np.array(self.objective_candidate, float).copy(),
            "layouts_x": self.x_candidate.copy(),
            "layouts_y": self.y_candidate.copy(),
        }
        self.history.append(event)

    def _run_optimization_generation(self):
        current_mode = self._get_current_evaluator_mode()

        self._current_generation_mode = current_mode
        self.eval_mode_log.append(current_mode)
        print(f"[Evaluator Mode] Generation {self.iteration_step + 1}: {current_mode}")

        prev_mode = getattr(self, "_population_mode", None)
        if prev_mode is None:
            prev_mode = self._get_init_evaluator_mode()
            self._population_mode = prev_mode

        if str(current_mode) != str(prev_mode):
            print(f"[MODE SWITCH] population_mode '{prev_mode}' -> '{current_mode}'")
            objective_before = self.objective_candidate.copy()

            self._rescore_population(current_mode)
            self._sort_population_by_objective()
            self._population_mode = current_mode

            print("[RESCORE DONE] population rescored & sorted under new mode.")
            print(f"  Mean AEP: {np.mean(objective_before)/1e9:.6f} -> {np.mean(self.objective_candidate)/1e9:.6f} GWh")
            print(f"  Max  AEP: {np.max(objective_before)/1e9:.6f} -> {np.max(self.objective_candidate)/1e9:.6f} GWh")

            self._append_history_event_mode_switch(
                prev_mode=prev_mode,
                new_mode=current_mode,
                objective_before=objective_before,
            )

        if self.random_seed is None:
            seeds = [None] * self.n_individuals
        else:
            base_seed = int(self.random_seed) + 55 + 1000 * max(self.iteration_step, 0)
            seeds = [base_seed + i for i in range(self.n_individuals)]

        multiargs = [
            (
                self.seconds_per_iteration,
                self.objective_candidate[i],
                self.x_candidate[i, :].copy(),
                self.y_candidate[i, :].copy(),
                self.fmodel_dict,
                self.wind_directions,
                self.wind_speeds,
                self.ti_table,
                self.freq_table,
                self.min_dist,
                self._boundary_polygon,
                self.distance_pmf,
                self.enable_geometric_yaw,
                seeds[i],
                self.p_no_wake_1d,
                self.lowrank_cut_in,
                self.eff_unity_ws_start,
                current_mode,
                self.precomputed_pivots,
                self.effective_n_ws,
                self.adaptive_k,
                self.tie_atol_x_max,
                self.adaptive_fill_strategy,
                self.reuse_stage1_overlap,
            )
            for i in range(self.n_individuals)
        ]

        if self._PoolExecutor:
            with self._PoolExecutor(self.max_workers) as p:
                out = p.starmap(run_single_individual_search, multiargs)
        else:
            out = [run_single_individual_search(*args) for args in multiargs]

        for i in range(self.n_individuals):
            self.objective_candidate[i] = out[i][0]
            self.x_candidate[i, :] = out[i][1]
            self.y_candidate[i, :] = out[i][2]
            self._num_objective_calls[i] = out[i][3]
            self._num_accept[i] = out[i][4]

        self.total_objective_calls += int(np.sum(self._num_objective_calls))

        self.objective_candidate_log.append(self.objective_candidate.copy())
        self.num_objective_calls_log.append(list(self._num_objective_calls))
        self.num_accept_log.append(list(self._num_accept))

        self._evaluate_opt_step()

    def _finalize_optimization(self):
        self.objective_final_search = float(self.objective_candidate[0])
        self.x_opt = self.x_candidate[0, :].copy()
        self.y_opt = self.y_candidate[0, :].copy()
        self.objective_final_exact = compute_aep_with_evaluator(
            layout_x=self.x_opt,
            layout_y=self.y_opt,
            fmodel_dict=self.fmodel_dict,
            wind_directions=self.wind_directions,
            wind_speeds=self.wind_speeds,
            ti_table=self.ti_table,
            freq_table=self.freq_table,
            p_no_wake_1d=self.p_no_wake_1d,
            lowrank_cut_in=self.lowrank_cut_in,
            eff_unity_ws_start=self.eff_unity_ws_start,
            mode="exact",
            precomputed_pivots=self.precomputed_pivots,
            effective_n_ws=self.effective_n_ws,
        )
        # Backward-compatible alias: objective_final is the search-evaluator value.
        self.objective_final = self.objective_final_search
        increase = 100.0 * (self.objective_final_search - self.objective_initial) / self.objective_initial
        print(
            f"Final search objective = {self.objective_final_search / 1e9:.6f} "
            f"{self._obj_unit} ({increase:+.2f}%)"
        )
        print(
            f"Final Exact AEP       = {self.objective_final_exact / 1e9:.6f} "
            f"{self._obj_unit}"
        )

    def optimize(self):
        self._initialize_optimization()
        while timerpc() < self._opt_stop_time:
            self._run_optimization_generation()
        self._finalize_optimization()
        return self.objective_final, self.x_opt, self.y_opt

    def describe(self):
        print("Low-Rank Random-Search Layout Optimization")
        print(f"Number of turbines = {self.N_turbines}")
        print(f"Minimum distance = {self.min_dist_D:.3f} [D]")
        print(f"Evaluator mode = {self.aep_mode}")
        print(f"Effective wind-speed bins = {self.effective_n_ws}")
        print(f"lowrank_cut_in = {self.lowrank_cut_in:.2f} m/s")
        print(f"rated_ws = {self.rated_ws:.2f} m/s")
        print(f"eff_unity_ws_start = {self.eff_unity_ws_start:.2f} m/s")
        print(f"pivot_ws_min/max = {self.pivot_ws_min:.2f} / {self.pivot_ws_max:.2f} m/s")
        if self.aep_mode == "hybrid":
            print(f"hybrid_exact_start_ratio = {self.hybrid_exact_start_ratio:.3f}")
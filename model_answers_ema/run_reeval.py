"""
run_reeval.py — Parallel robustness re-evaluation for Assignment 8
==================================================================
Runs the full policy × scenario experiment using EMA Workbench's
MultiprocessingEvaluator, which is not available inside a Jupyter notebook
(the 'spawn' start method cannot pickle interactively-defined functions).

Results are saved in the same cache format as the notebook so they are
loaded automatically when you open Assignment 8.

Usage
-----
  # From the repo root (where JUSTICE-main/ lives):
  cd /path/to/epa141a

  # Default: 28 policies × 1000 FAIR scenarios, all CPU cores
  .venv/bin/python model_answers_ema/run_reeval.py

  # Custom number of scenarios and cores
  .venv/bin/python model_answers_ema/run_reeval.py --n_scenarios 200 --n_cores 4

  # Quick smoke-test (5 scenarios, 2 cores)
  .venv/bin/python model_answers_ema/run_reeval.py --n_scenarios 5 --n_cores 2

Runtime estimate (sequential time ÷ n_cores)
  50  scenarios →  ~1 400 calls → ~20–40 min on 1 core → ~5–10 min on 4 cores
  200 scenarios →  ~5 600 calls → ~80–160 min on 1 core → ~20–40 min on 4 cores
  1000 scenarios → ~28 000 calls → ~7–14 h on 1 core   → ~2–4 h on 4 cores
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# ── Paths (relative to this file's location) ──────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_JUSTICE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../JUSTICE-main"))
_CONFIG_DIR   = os.path.normpath(os.path.join(_SCRIPT_DIR, "../config"))
RESULTS_ROOT  = os.path.join(_SCRIPT_DIR, "results")

if _JUSTICE_ROOT not in sys.path:
    sys.path.insert(0, _JUSTICE_ROOT)

# JUSTICE data loaders resolve paths relative to the JUSTICE root.
os.chdir(_JUSTICE_ROOT)

# ── JUSTICE imports ────────────────────────────────────────────────────────────
from justice.model import JUSTICE
from justice.util.data_loader import DataLoader
from justice.util.enumerations import Abatement, DamageFunction, Economy, WelfareFunction
from justice.util.emission_control_constraint import EmissionControlConstraint
from justice.util.model_time import TimeHorizon
from justice.objectives.objective_functions import years_above_temperature_threshold
from solvers.emodps.rbf import RBF

# ── Config ────────────────────────────────────────────────────────────────────
with open(os.path.join(_CONFIG_DIR, "config_ssp245.json")) as _fh:
    _cfg = json.load(_fh)

_time_horizon = TimeHorizon(
    start_year   = _cfg["start_year"],
    end_year     = _cfg["end_year"],
    data_timestep= _cfg["data_timestep"],
    timestep     = _cfg["timestep"],
)

N_TIMESTEPS = len(_time_horizon.model_time_horizon)
N_REGIONS   = len(DataLoader().REGION_LIST)
N_INPUTS    = _cfg["n_inputs"]
N_RBFS      = N_INPUTS + 2
SCENARIO    = _cfg["reference_ssp_rcp_scenario_index"]
EC_START_TS = _time_horizon.year_to_timestep(
    year     = _cfg["emission_control_start_year"],
    timestep = _cfg["timestep"],
)

_rbf_tmp          = RBF(n_rbfs=N_RBFS, n_inputs=N_INPUTS, n_outputs=N_REGIONS)
C_SHAPE, R_SHAPE, W_SHAPE = _rbf_tmp.get_shape()

_MAX_TEMP, _MIN_TEMP = 16.0, 0.0
_MAX_DIFF, _MIN_DIFF =  2.0, 0.0

OBJECTIVES = ["welfare", "years_above_2C", "welfare_loss_damage", "welfare_loss_abatement"]


# ── Model wrapper (must be at module level for multiprocessing) ────────────────
def model_wrapper_reeval(**kwargs) -> tuple:
    """
    Called once per (policy, scenario) pair by EMA Workbench perform_experiments.
    kwargs contains lever values (RBF parameters) + climate_ensemble_index.
    """
    ensemble_index = int(kwargs.pop("climate_ensemble_index"))

    rbf     = RBF(n_rbfs=N_RBFS, n_inputs=N_INPUTS, n_outputs=N_REGIONS)
    centers = np.array([kwargs.pop(f"center {i}") for i in range(C_SHAPE[0])])
    radii   = np.array([kwargs.pop(f"radii {i}")  for i in range(R_SHAPE[0])])
    weights = np.array([kwargs.pop(f"weights {i}") for i in range(W_SHAPE[0])])
    rbf.set_decision_vars(np.concatenate([centers, radii, weights]))

    constraint = EmissionControlConstraint(
        max_annual_growth_rate          = 0.04,
        emission_control_start_timestep = EC_START_TS,
        min_emission_control_rate       = 0.01,
    )

    model = JUSTICE(
        scenario                    = SCENARIO,
        climate_ensembles           = [ensemble_index],
        economy_type                = Economy.NEOCLASSICAL,
        damage_function_type        = DamageFunction.KALKUHL,
        abatement_type              = Abatement.ENERDATA,
        social_welfare_function_type= WelfareFunction.UTILITARIAN.value[0],
    )
    no_ens          = model.no_of_ensembles   # 1
    ecr             = np.zeros((N_REGIONS, N_TIMESTEPS, no_ens))
    constrained_ecr = np.zeros_like(ecr)
    prev_temp       = np.zeros(no_ens)
    diff            = np.zeros(no_ens)

    for t in range(N_TIMESTEPS):
        constrained_ecr[:, t, :] = constraint.constrain_emission_control_rate(
            ecr[:, t, :], t, allow_fallback=False
        )
        model.stepwise_run(
            emission_control_rate  = constrained_ecr[:, t, :],
            timestep               = t,
            endogenous_savings_rate= True,
        )
        data_t = model.stepwise_evaluate(timestep=t)
        temp   = data_t["global_temperature"][t, :]

        if t % 5 == 0:
            diff      = temp - prev_temp
            prev_temp = temp.copy()

        scaled_temp = (temp - _MIN_TEMP) / (_MAX_TEMP - _MIN_TEMP)
        scaled_diff = (diff - _MIN_DIFF) / (_MAX_DIFF - _MIN_DIFF)

        if t < N_TIMESTEPS - 1:
            ecr[:, t + 1, :] = rbf.apply_rbfs(np.array([scaled_temp, scaled_diff]))

    data = model.evaluate()

    welfare = float(np.abs(data["welfare"]))
    welfare = welfare if np.isfinite(welfare) else 1e6

    yrs_above = float(
        years_above_temperature_threshold(data["global_temperature"], threshold=2.0)
    )

    _, _, _, wl_damage = model.welfare_function.calculate_welfare(
        data["damage_cost_per_capita"], welfare_loss=True
    )
    wl_damage = float(np.abs(wl_damage)) if np.isfinite(wl_damage) else 1e6

    _, _, _, wl_abatement = model.welfare_function.calculate_welfare(
        data["abatement_cost_per_capita"], welfare_loss=True
    )
    wl_abatement = float(np.abs(wl_abatement)) if np.isfinite(wl_abatement) else 1e6

    return (welfare, yrs_above, wl_damage, wl_abatement)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel robustness re-evaluation for JUSTICE.")
    parser.add_argument("--n_scenarios", type=int, default=1000,
                        help="Number of FAIR ensemble members to use (default: 1000 = full ensemble)")
    parser.add_argument("--n_cores",    type=int, default=None,
                        help="Number of CPU cores (default: all available)")
    args = parser.parse_args()

    N_SCENARIOS      = args.n_scenarios
    SCENARIO_INDICES = list(np.linspace(1, 1000, N_SCENARIOS, dtype=int))

    # ── Load reference set ─────────────────────────────────────────────────────
    ref_path = os.path.join(RESULTS_ROOT, "reference_set_utilitarian.csv")
    if not os.path.exists(ref_path):
        ref_path = os.path.join(RESULTS_ROOT, "UTILITARIAN_reference_set.csv")
        print(f"Grand reference set not found — falling back to {ref_path}")

    ref_set  = pd.read_csv(ref_path)
    ref_set  = ref_set[ref_set["welfare"] < 1e5].reset_index(drop=True)
    OPT_OBJECTIVES = ["welfare", "fraction_above_threshold",
                      "welfare_loss_damage", "welfare_loss_abatement"]
    LEVER_COLS = [c for c in ref_set.columns if c not in OPT_OBJECTIVES]

    N_POLICIES   = len(ref_set)
    N_OBJECTIVES = len(OBJECTIVES)

    RESULTS_PATH     = os.path.join(RESULTS_ROOT,
                                    f"reeval_utilitarian_{N_POLICIES}p_{N_SCENARIOS}s.npy")
    EXPERIMENTS_PATH = os.path.join(RESULTS_ROOT,
                                    f"reeval_utilitarian_{N_POLICIES}p_{N_SCENARIOS}s_experiments.csv")

    print(f"Policies  : {N_POLICIES}")
    print(f"Scenarios : {N_SCENARIOS}  (FAIR indices: {SCENARIO_INDICES[:3]} … {SCENARIO_INDICES[-3:]})")
    print(f"Cache     : {RESULTS_PATH}")

    if os.path.exists(RESULTS_PATH) and os.path.exists(EXPERIMENTS_PATH):
        print("Cache already exists — delete it to force a rerun:")
        print(f"  rm {RESULTS_PATH} {EXPERIMENTS_PATH}")
        sys.exit(0)

    # ── Build EMA objects ──────────────────────────────────────────────────────
    from ema_workbench import (
        Model, RealParameter, IntegerParameter, ScalarOutcome,
        Policy, Scenario, MultiprocessingEvaluator, ema_logging,
    )
    ema_logging.log_to_stderr(ema_logging.INFO)

    ema_model = Model("JUSTICEreeval", function=model_wrapper_reeval)
    ema_model.uncertainties = [IntegerParameter("climate_ensemble_index", 1, 1000)]

    n_cr = C_SHAPE[0]
    n_w  = W_SHAPE[0]
    ema_model.levers = (
        [RealParameter(f"center {i}", -1.0, 1.0) for i in range(n_cr)]
        + [RealParameter(f"radii {i}",  0.0, 1.0) for i in range(n_cr)]
        + [RealParameter(f"weights {i}", 0.0, 1.0) for i in range(n_w)]
    )
    ema_model.outcomes = [
        ScalarOutcome("welfare",                kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("years_above_2C",         kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("welfare_loss_damage",    kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("welfare_loss_abatement", kind=ScalarOutcome.MINIMIZE),
    ]

    policies  = [
        Policy(f"P{pi}", **{col: float(ref_set.iloc[pi][col]) for col in LEVER_COLS})
        for pi in range(N_POLICIES)
    ]
    scenarios = [
        Scenario(f"FAIR_{idx}", climate_ensemble_index=int(idx))
        for idx in SCENARIO_INDICES
    ]

    # ── Run ────────────────────────────────────────────────────────────────────
    n_cores_msg = args.n_cores if args.n_cores else "all available"
    print(f"\nStarting MultiprocessingEvaluator with {n_cores_msg} cores …")

    with MultiprocessingEvaluator(ema_model, n_processes=args.n_cores) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=scenarios,
            policies=policies,
        )

    # ── Reshape and save ───────────────────────────────────────────────────────
    policy_name_to_idx   = {f"P{pi}": pi for pi in range(N_POLICIES)}
    scenario_name_to_idx = {f"FAIR_{idx}": si for si, idx in enumerate(SCENARIO_INDICES)}

    results = np.full((N_POLICIES, N_SCENARIOS, N_OBJECTIVES), np.nan)
    for row_i, row in experiments.iterrows():
        pi = policy_name_to_idx.get(row["policy"])
        si = scenario_name_to_idx.get(row["scenario"])
        if pi is None or si is None:
            continue
        for oi, obj in enumerate(OBJECTIVES):
            results[pi, si, oi] = outcomes[obj][row_i]

    np.save(RESULTS_PATH, results)
    experiments.to_csv(EXPERIMENTS_PATH, index=False)

    print(f"\nDone. Results shape: {results.shape}")
    print(f"NaN entries: {np.isnan(results).sum()}")
    print(f"Saved to: {RESULTS_PATH}")
    print(f"          {EXPERIMENTS_PATH}")
    print("\nOpen Assignment 8 — it will load these results automatically.")

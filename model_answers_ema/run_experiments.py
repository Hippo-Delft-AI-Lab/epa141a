"""
run_experiments.py — Assignment 1: Exploratory Modelling with JUSTICE
======================================================================
Runs JUSTICE across a Latin-Hypercube ensemble using MultiprocessingEvaluator,
then saves results to CSV so the notebook can load them without re-running.

Why a script instead of a notebook cell?
-----------------------------------------
On macOS and Windows, Python's multiprocessing uses the 'spawn' start method,
which requires all worker-process code to be importable at module level and
protected by  ``if __name__ == '__main__':``.  Jupyter notebook cells have no
such guard, so MultiprocessingEvaluator can fail or hang inside a kernel.
Running from a terminal script sidesteps this entirely.

Usage
-----
From the ``model_answers_ema/`` directory:

    # Quick smoke-test — 20 scenarios, 1 worker (< 2 min)
    python run_experiments.py --scenarios 20 --n_processes 1

    # Default run — 500 scenarios across all CPU cores (~5-15 min)
    python run_experiments.py

    # Custom scenario count and output directory
    python run_experiments.py --scenarios 1000 --output_dir results/

    # Background run with logged output (macOS / Linux)
    nohup python run_experiments.py --scenarios 1000 > experiments_log.txt 2>&1 &

Arguments
---------
--scenarios     int     Number of LHS scenarios per policy (default: 500)
--n_processes   int     Worker processes; 0 or omit → cpu_count - 1 (default: auto)
--output_dir    str     Directory for output CSVs (default: model_answers_ema/results/)
--policies      str+    Policy names to run; choices: no_abatement moderate_abatement
                        (default: both)

Outputs (written to --output_dir)
----------------------------------
experiments.csv             One row per run; parameter values + policy label.
outcomes_scalar.csv         Scalar outcomes aligned row-by-row with experiments.csv.
outcomes_temperature.npy    NumPy array (n_runs × n_timesteps) of temperature
                            trajectories, aligned with experiments.csv.

Loading results in the notebook
---------------------------------
    import numpy as np, pandas as pd

    experiments = pd.read_csv('results/experiments.csv')
    outcomes_df = pd.read_csv('results/outcomes_scalar.csv')
    temperature = np.load('results/outcomes_temperature.npy')
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — locate JUSTICE-main relative to this script's directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_JUSTICE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "../JUSTICE-main"))

if not os.path.isdir(_JUSTICE_ROOT):
    raise FileNotFoundError(
        f"JUSTICE-main not found at: {_JUSTICE_ROOT}\n"
        "Ensure run_experiments.py lives in model_answers_ema/ and "
        "JUSTICE-main/ is its sibling directory."
    )

if _JUSTICE_ROOT not in sys.path:
    sys.path.insert(0, _JUSTICE_ROOT)

os.chdir(_JUSTICE_ROOT)

# ---------------------------------------------------------------------------
# EMA Workbench + JUSTICE imports (after path setup)
# ---------------------------------------------------------------------------
import logging

from ema_workbench import (
    ArrayOutcome,
    Model,
    MultiprocessingEvaluator,
    RealParameter,
    Sample,
    ScalarOutcome,
    ema_logging,
)
from justice.model import JUSTICE
from justice.objectives.objective_functions import years_above_temperature_threshold
from justice.util.enumerations import WelfareFunction

ema_logging.log_to_stderr(logging.INFO)

# ---------------------------------------------------------------------------
# Model function
# ---------------------------------------------------------------------------

def justice_model(rho=0.015, eta=1.45, delta=1.0, ecs_ensemble=1, ecr_plateau=0.0):
    """EMA Workbench function model — configurable abatement, Utilitarian welfare.

    Parameters
    ----------
    rho : float
        Pure rate of social time preference (Ramsey discount rate).
    eta : float
        Elasticity of marginal utility of consumption.
    delta : float
        Damage function scaling factor (1.0 = default; >1 = more damage).
    ecs_ensemble : float
        FAIR climate ensemble index in [1, 1001].
    ecr_plateau : float
        Emission control rate applied uniformly across all regions and timesteps.
        0.0 = no abatement; 0.4 = 40 % abatement (moderate policy).
    """
    JUSTICE.hard_reset()
    ensemble_idx = int(np.round(np.clip(ecs_ensemble, 1, 1001)))
    model = JUSTICE(
        start_year=2015, end_year=2300, timestep=1,
        scenario=2, climate_ensembles=ensemble_idx, stochastic_run=False,
        social_welfare_function=WelfareFunction.UTILITARIAN,
    )
    model.economy.pure_rate_of_social_time_preference               = float(rho)
    model.economy.elasticity_of_marginal_utility_of_consumption     = float(eta)
    model.welfare_function.pure_rate_of_social_time_preference      = float(rho)
    model.welfare_function.elasticity_of_marginal_utility_of_consumption = float(eta)
    model.damage_function.coefficient_a                  *= float(delta)
    model.damage_function.coefficient_b                  *= float(delta)
    model.damage_function.damage_gdp_ratio_with_gradient *= float(delta)

    ecr = np.full(model.emission_control_rate.shape[:2], float(ecr_plateau))
    model.run(emission_control_rate=ecr, endogenous_savings_rate=True)
    datasets = model.evaluate()

    welfare = float(np.abs(np.squeeze(datasets["welfare"])))
    yat     = float(np.squeeze(
        years_above_temperature_threshold(datasets["global_temperature"], 2.0)
    ))
    _, _, _, wl_dmg = model.welfare_function.calculate_welfare(
        datasets["damage_cost_per_capita"], welfare_loss=True)
    _, _, _, wl_abt = model.welfare_function.calculate_welfare(
        datasets["abatement_cost_per_capita"], welfare_loss=True)

    temp = np.squeeze(datasets["global_temperature"])
    if temp.ndim == 2:
        temp = temp.mean(axis=0)

    return {
        "welfare":                           welfare,
        "years_above_temperature_threshold": yat,
        "welfare_loss_damage":               float(np.abs(np.squeeze(wl_dmg))),
        "welfare_loss_abatement":            float(np.abs(np.squeeze(wl_abt))),
        "temperature_trajectory":            temp.astype(float),
    }


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def build_model() -> Model:
    em_model = Model("JUSTICE", function=justice_model)

    em_model.uncertainties = [
        RealParameter("rho",          0.001,    0.030),
        RealParameter("eta",          0.5,      1.5),
        RealParameter("delta",        0.5,      2.0),
        RealParameter("ecs_ensemble", 1,     1001),
    ]
    em_model.levers = [
        RealParameter("ecr_plateau", 0.0, 1.0),
    ]
    em_model.outcomes = [
        ScalarOutcome("welfare"),
        ScalarOutcome("years_above_temperature_threshold"),
        ScalarOutcome("welfare_loss_damage"),
        ScalarOutcome("welfare_loss_abatement"),
        ArrayOutcome("temperature_trajectory"),
    ]
    return em_model


ALL_POLICIES = {
    "no_abatement":       Sample("no_abatement",       ecr_plateau=0.0),
    "moderate_abatement": Sample("moderate_abatement", ecr_plateau=0.4),
}

SCALAR_OUTCOMES = [
    "welfare",
    "years_above_temperature_threshold",
    "welfare_loss_damage",
    "welfare_loss_abatement",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments.py",
        description="Run JUSTICE exploratory experiments with MultiprocessingEvaluator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenarios",   type=int, default=500,
                   help="Number of LHS scenarios per policy")
    p.add_argument("--n_processes", type=int, default=None,
                   help="Worker processes (None = cpu_count - 1)")
    p.add_argument("--output_dir",  type=str,
                   default=os.path.join(_SCRIPT_DIR, "results"),
                   help="Output directory for CSV and NPY files")
    p.add_argument("--policies",    type=str, nargs="+",
                   default=list(ALL_POLICIES.keys()),
                   choices=list(ALL_POLICIES.keys()),
                   help="Which policies to run")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    policies = [ALL_POLICIES[name] for name in args.policies]

    print("=" * 60)
    print("JUSTICE — Assignment 1 Exploratory Experiments")
    print("=" * 60)
    print(f"  Scenarios per policy : {args.scenarios}")
    print(f"  Policies             : {args.policies}")
    print(f"  Total runs           : {args.scenarios * len(policies)}")
    print(f"  Worker processes     : {args.n_processes or 'auto (cpu_count-1)'}")
    print(f"  Output directory     : {output_dir}")
    print("=" * 60)

    em_model = build_model()

    t0 = time.time()
    with MultiprocessingEvaluator(em_model, n_processes=args.n_processes) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=args.scenarios, policies=policies
        )
    elapsed = time.time() - t0
    print(f"\nCompleted {len(experiments)} runs in {elapsed / 60:.1f} min.")

    # -- Save experiments (parameters + policy label) ----------------------
    exp_path = os.path.join(output_dir, "experiments.csv")
    experiments.to_csv(exp_path, index=False)
    print(f"  experiments      → {exp_path}")

    # -- Save scalar outcomes ----------------------------------------------
    scalar_df = pd.DataFrame({k: outcomes[k] for k in SCALAR_OUTCOMES})
    scalar_df["policy"] = experiments["policy"].values
    scalar_path = os.path.join(output_dir, "outcomes_scalar.csv")
    scalar_df.to_csv(scalar_path, index=False)
    print(f"  scalar outcomes  → {scalar_path}")

    # -- Save temperature trajectories -------------------------------------
    temp_arr = np.stack(outcomes["temperature_trajectory"])
    temp_path = os.path.join(output_dir, "outcomes_temperature.npy")
    np.save(temp_path, temp_arr)
    print(f"  temperature      → {temp_path}  (shape {temp_arr.shape})")

    print("\nDone. Load in the notebook with:")
    print(f"  experiments = pd.read_csv('{output_dir}/experiments.csv')")
    print(f"  outcomes_df = pd.read_csv('{output_dir}/outcomes_scalar.csv')")
    print(f"  temperature = np.load('{output_dir}/outcomes_temperature.npy')")


if __name__ == "__main__":
    main()

"""
run_optimization_local.py  —  Assignment 5 Model Answer
=========================================================
Local multi-objective optimisation for JUSTICE using GenerationalBorg
(ema_workbench / Platypus).

Optimises 244 RBF policy parameters under the **Utilitarian** social welfare
function, tracking 4 objectives:
  1. welfare                  [MINIMIZE]
  2. fraction_above_threshold [MINIMIZE]  fraction of ensemble members whose
                                          temperature exceeds 2 °C in 2100
  3. welfare_loss_damage      [MAXIMIZE]
  4. welfare_loss_abatement   [MAXIMIZE]

Local-run adaptations (vs the HPC runner in analysis/hpc_runner_student.py):
  - Uses 10 well-distributed FAIR ensemble members instead of all 1000.
    This gives ~30–100x speedup per evaluation, making the run feasible on
    a laptop or desktop.
  - Uses MultiprocessingEvaluator (shared-memory) instead of MPIEvaluator.
  - Seeds are run sequentially; each seed uses all available CPU cores.

Usage (from any directory — the script finds JUSTICE-main automatically):
    python run_optimization_local.py                   # all defaults
    python run_optimization_local.py --nfe 500         # quick smoke-test
    python run_optimization_local.py --seeds 9845531   # one seed only
    python run_optimization_local.py --n_processes 4   # limit to 4 cores

Outputs written to <output_dir>/ (default: model_answers_ema/results/):
    UTILITARIAN_<nfe>_<seed>/
        UTILITARIAN_<nfe>_<seed>.tar.gz   convergence archive (ArchiveLogger)
        pareto_front_<seed>.csv           final Pareto-optimal solutions

Feasibility guidance
---------------------
With n_ensembles=10 and 8 CPU cores, one 50 000-NFE seed takes roughly 1-3 h.
All five seeds therefore finish in ~1 working day.

Recommended workflow:
  1. Quick test (< 5 min):
         python run_optimization_local.py --nfe 500
  2. Single-seed overnight run:
         python run_optimization_local.py --nfe 50000 --seeds 9845531
  3. Full 5-seed run (background):
         nohup python run_optimization_local.py --nfe 50000 > opt_log.txt 2>&1 &
     or with screen:
         screen -S justice; python run_optimization_local.py --nfe 50000
         # Ctrl-A D to detach, screen -r justice to reattach
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — all paths are derived from the repo root so that JUSTICE-main
# remains completely unmodified (treated as a read-only dependency).
#
#   epa141a/              ← _REPO_ROOT
#   ├── JUSTICE-main/     ← _JUSTICE_ROOT   (never written to)
#   ├── config/           ← _CONFIG_DIR     (course config files)
#   └── model_answers_ema/  ← this script
#       └── results/      ← default output  (generated artifacts)
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT    = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_JUSTICE_ROOT = os.path.normpath(os.path.join(_REPO_ROOT, "JUSTICE-main"))
_CONFIG_DIR   = os.path.normpath(os.path.join(_REPO_ROOT, "config"))

if not os.path.isdir(_JUSTICE_ROOT):
    raise FileNotFoundError(
        f"JUSTICE-main not found at: {_JUSTICE_ROOT}\n"
        "Ensure this script lives in model_answers_ema/ and JUSTICE-main/ "
        "is its sibling directory."
    )

if _JUSTICE_ROOT not in sys.path:
    sys.path.insert(0, _JUSTICE_ROOT)

# Change CWD so relative data-file paths inside JUSTICE resolve correctly.
os.chdir(_JUSTICE_ROOT)

# ---------------------------------------------------------------------------
# EMA Workbench
# ---------------------------------------------------------------------------
from ema_workbench import (  # noqa: E402
    Constant,
    Model,
    RealParameter,
    ScalarOutcome,
    CategoricalParameter,
    MultiprocessingEvaluator,
    Sample,
    ema_logging,
)
from ema_workbench.em_framework.optimization import (  # noqa: E402
    GenerationalBorg,
)

# ---------------------------------------------------------------------------
# JUSTICE
# ---------------------------------------------------------------------------
from justice.model import JUSTICE  # noqa: E402
from justice.util.data_loader import DataLoader  # noqa: E402
from justice.util.enumerations import (  # noqa: E402
    Abatement,
    DamageFunction,
    Economy,
    WelfareFunction,
)
from justice.util.emission_control_constraint import EmissionControlConstraint  # noqa: E402
from justice.util.model_time import TimeHorizon  # noqa: E402
from justice.objectives.objective_functions import (  # noqa: E402
    fraction_of_ensemble_above_threshold,
)
from solvers.emodps.rbf import RBF  # noqa: E402

# ---------------------------------------------------------------------------
# Scaling constants — identical to EMA_model_wrapper.py
# ---------------------------------------------------------------------------
_MAX_TEMP = 16.0
_MIN_TEMP = 0.0
_MAX_DIFF = 2.0
_MIN_DIFF = 0.0


# ===========================================================================
# Model wrapper
# ===========================================================================

def model_wrapper_local(**kwargs) -> tuple[float, float, float, float]:
    """
    JUSTICE evaluation function for ema_workbench optimisation.

    Mirrors ``justice.util.EMA_model_wrapper.model_wrapper_emodps`` with two
    changes:
      * Accepts ``climate_ensemble_indices`` (list[int]) as a Constant so that
        only the specified FAIR members are loaded.
      * Explicitly pops every Constant from kwargs (no leftover keys).

    Return order must match ``ema_model.outcomes`` defined in ``run_seed()``:
        (welfare, fraction_above_threshold,
         welfare_loss_damage, welfare_loss_abatement)
    """
    # -- constants ----------------------------------------------------------
    scenario = kwargs.pop("ssp_rcp_scenario")
    swf_type = kwargs.pop("social_welfare_function_type")

    economy_type         = Economy.from_index(kwargs.pop("economy_type"))
    damage_function_type = DamageFunction.from_index(kwargs.pop("damage_function_type"))
    abatement_type       = Abatement.from_index(kwargs.pop("abatement_type"))

    n_regions                       = kwargs.pop("n_regions")
    n_timesteps                     = kwargs.pop("n_timesteps")
    emission_control_start_timestep = kwargs.pop("emission_control_start_timestep")
    n_inputs_rbf                    = kwargs.pop("n_inputs_rbf")
    n_outputs_rbf                   = kwargs.pop("n_outputs_rbf")
    temp_year_idx                   = kwargs.pop("temperature_year_of_interest_index")
    ensemble_indices                = kwargs.pop("climate_ensemble_indices")

    # -- RBF ----------------------------------------------------------------
    rbf = RBF(
        n_rbfs=(n_inputs_rbf + 2),   # rule of thumb used throughout JUSTICE
        n_inputs=n_inputs_rbf,
        n_outputs=n_outputs_rbf,
    )
    c_shape, r_shape, w_shape = rbf.get_shape()

    centers = np.array([kwargs.pop(f"center {i}")  for i in range(c_shape[0])])
    radii   = np.array([kwargs.pop(f"radii {i}")   for i in range(r_shape[0])])
    weights = np.array([kwargs.pop(f"weights {i}") for i in range(w_shape[0])])
    rbf.set_decision_vars(np.concatenate([centers, radii, weights]))

    # -- emission ramp-up constraint ----------------------------------------
    constraint = EmissionControlConstraint(
        max_annual_growth_rate=0.04,
        emission_control_start_timestep=emission_control_start_timestep,
        min_emission_control_rate=0.01,
    )

    # -- per-process singleton JUSTICE instance -----------------------------
    # Each worker process (spawned by MultiprocessingEvaluator) initialises its
    # own JUSTICE model once, then reuses it via reset_model().
    if not hasattr(model_wrapper_local, "_instance"):
        model_wrapper_local._instance = JUSTICE(
            scenario=scenario,
            climate_ensembles=list(ensemble_indices),
            economy_type=economy_type,
            damage_function_type=damage_function_type,
            abatement_type=abatement_type,
            social_welfare_function_type=swf_type,
        )
    else:
        model_wrapper_local._instance.reset_model()

    model           = model_wrapper_local._instance
    no_of_ensembles = model.no_of_ensembles

    # -- stepwise simulation -----------------------------------------------
    ecr             = np.zeros((n_regions, n_timesteps, no_of_ensembles))
    constrained_ecr = np.zeros_like(ecr)
    prev_temp       = 0.0
    diff            = 0.0

    for t in range(n_timesteps):
        constrained_ecr[:, t, :] = constraint.constrain_emission_control_rate(
            ecr[:, t, :], t, allow_fallback=False
        )
        model.stepwise_run(
            emission_control_rate=constrained_ecr[:, t, :],
            timestep=t,
            endogenous_savings_rate=True,
        )
        data = model.stepwise_evaluate(timestep=t)
        temp = data["global_temperature"][t, :]

        if t % 5 == 0:
            diff      = temp - prev_temp
            prev_temp = temp

        scaled_temp = (temp - _MIN_TEMP) / (_MAX_TEMP - _MIN_TEMP)
        scaled_diff = (diff - _MIN_DIFF) / (_MAX_DIFF - _MIN_DIFF)

        if t < n_timesteps - 1:
            ecr[:, t + 1, :] = rbf.apply_rbfs(np.array([scaled_temp, scaled_diff]))

    # -- final objectives ---------------------------------------------------
    data = model.evaluate()

    welfare = float(np.abs(data["welfare"]))
    welfare = welfare if np.isfinite(welfare) else 1e6  # penalty for NaN/Inf

    frac = fraction_of_ensemble_above_threshold(
        temperature=data["global_temperature"],
        temperature_year_index=temp_year_idx,
        threshold=2.0,
    )
    frac = float(frac) if np.isfinite(float(frac)) else 1.0  # worst-case fraction

    _, _, _, wl_damage = model.welfare_function.calculate_welfare(
        data["damage_cost_per_capita"], welfare_loss=True
    )
    wl_damage = float(np.abs(wl_damage))
    wl_damage = wl_damage if np.isfinite(wl_damage) else 0.0  # worst-case for MAXIMIZE

    _, _, _, wl_abatement = model.welfare_function.calculate_welfare(
        data["abatement_cost_per_capita"], welfare_loss=True
    )
    wl_abatement = float(np.abs(wl_abatement))
    wl_abatement = wl_abatement if np.isfinite(wl_abatement) else 0.0  # worst-case for MAXIMIZE

    return (welfare, frac, wl_damage, wl_abatement)


# ===========================================================================
# Single-seed optimisation
# ===========================================================================

def run_seed(
    config_path: str,
    seed: int,
    nfe: int,
    output_dir: str,
    n_ensembles: int = 10,
    population_size: int = 100,
    n_processes: int | None = None,
):
    """
    Run one optimisation seed with MultiprocessingEvaluator.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file (absolute, or relative to CWD).
    seed : int
        Random seed.
    nfe : int
        Number of function evaluations.
    output_dir : str
        Root results directory (absolute path recommended).
    n_ensembles : int
        Number of FAIR ensemble members to use (default 10).
    population_size : int
        MOEA population size (default 100).
    n_processes : int or None
        Worker count; None → cpu_count - 1.

    Returns
    -------
    results : pandas.DataFrame
        Pareto-optimal solutions (levers + objectives).
    """
    # -- load config -------------------------------------------------------
    with open(config_path) as fh:
        cfg = json.load(fh)

    start_year                   = cfg["start_year"]
    end_year                     = cfg["end_year"]
    data_timestep                = cfg["data_timestep"]
    timestep                     = cfg["timestep"]
    emission_control_start_year  = cfg["emission_control_start_year"]
    n_inputs                     = cfg["n_inputs"]
    epsilons                     = cfg["epsilons"]
    temp_year_of_interest        = cfg["temperature_year_of_interest"]
    reference_scenario_idx       = cfg["reference_ssp_rcp_scenario_index"]

    # -- derived quantities ------------------------------------------------
    data_loader   = DataLoader()
    n_regions     = len(data_loader.REGION_LIST)   # 57
    n_rbfs_actual = n_inputs + 2                   # 4

    time_horizon = TimeHorizon(
        start_year=start_year,
        end_year=end_year,
        data_timestep=data_timestep,
        timestep=timestep,
    )
    n_timesteps = len(time_horizon.model_time_horizon)   # 286

    ec_start_ts = time_horizon.year_to_timestep(
        year=emission_control_start_year, timestep=timestep
    )
    temp_year_idx = time_horizon.year_to_timestep(
        year=temp_year_of_interest, timestep=timestep
    )

    # -- 10 well-distributed FAIR ensemble members -------------------------
    # JUSTICE uses 1-based ensemble indexing (asserts >= 1, subtracts 1 internally)
    # np.linspace(1, 1000, 10, dtype=int) → [1, 112, 223, 334, 445,
    #                                         556, 667, 778, 889, 1000]
    ensemble_indices = list(np.linspace(1, 1000, n_ensembles, dtype=int))

    # -- EMA model ---------------------------------------------------------
    ema_model = Model("JUSTICE", function=model_wrapper_local)

    #  Constants: fixed values passed to every model evaluation
    ema_model.constants = [
        Constant("n_regions",                        n_regions),
        Constant("n_timesteps",                      n_timesteps),
        Constant("emission_control_start_timestep",  ec_start_ts),
        Constant("n_rbfs",                           n_rbfs_actual),
        Constant("n_inputs_rbf",                     n_inputs),
        Constant("n_outputs_rbf",                    n_regions),
        Constant("social_welfare_function_type",
                 WelfareFunction.UTILITARIAN.value[0]),   # 0
        Constant("economy_type",         Economy.NEOCLASSICAL.value),   # 0
        Constant("damage_function_type", DamageFunction.KALKUHL.value), # 1
        Constant("abatement_type",       Abatement.ENERDATA.value),     # 0
        Constant("temperature_year_of_interest_index", temp_year_idx),
        Constant("climate_ensemble_indices", ensemble_indices),
    ]

    #  Uncertainties: the optimisation is performed under a single reference
    #  scenario (SSP2-RCP4.5), but the CategoricalParameter is kept so the
    #  model signature is identical to the HPC version.
    ema_model.uncertainties = [
        CategoricalParameter("ssp_rcp_scenario", tuple(range(8))),
    ]

    #  Levers: 244 continuous RBF parameters
    #    centers  (8):  range [-1,  1]
    #    radii    (8):  range [ 0,  1]
    #    weights (228): range [ 0,  1]
    n_cr = n_rbfs_actual * n_inputs   # centers = radii = 8
    n_w  = n_rbfs_actual * n_regions  # weights = 228

    ema_model.levers = (
        [RealParameter(f"center {i}",  -1.0, 1.0) for i in range(n_cr)]
        + [RealParameter(f"radii {i}",   0.0, 1.0) for i in range(n_cr)]
        + [RealParameter(f"weights {i}", 0.0, 1.0) for i in range(n_w)]
    )

    #  Outcomes: positional order matches model_wrapper_local return tuple
    ema_model.outcomes = [
        ScalarOutcome("welfare",                  kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("fraction_above_threshold", kind=ScalarOutcome.MINIMIZE),
        ScalarOutcome("welfare_loss_damage",      kind=ScalarOutcome.MAXIMIZE),
        ScalarOutcome("welfare_loss_abatement",   kind=ScalarOutcome.MAXIMIZE),
    ]

    # -- output setup ------------------------------------------------------
    seed_dir = os.path.join(output_dir, f"UTILITARIAN_{nfe}_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    archive_filename = f"UTILITARIAN_{nfe}_{seed}.tar.gz"

    # ema_workbench 3.0 raises FileExistsError if archive already exists.
    archive_path = os.path.join(seed_dir, archive_filename)
    if os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"    removed stale archive: {archive_path}")

    reference = Sample("reference", ssp_rcp_scenario=reference_scenario_idx)

    # -- run ---------------------------------------------------------------
    # ema_workbench 3.0 API: archive stored via filename/directory params;
    # convergence DataFrame returned directly as second element of each tuple.
    # ArchiveLogger, EpsilonProgress, OperatorProbabilities removed in 3.0.
    print(f"    workers = {n_processes if n_processes else 'auto (cpu_count-1)'}")
    with MultiprocessingEvaluator(ema_model, n_processes=n_processes) as evaluator:
        run_results = evaluator.optimize(
            searchover="levers",
            nfe=nfe,
            epsilons=epsilons,
            reference=reference,
            filename=archive_filename,
            directory=seed_dir,
            population_size=population_size,
            algorithm=GenerationalBorg,
        )

    # ema_workbench 3.0 returns list[tuple[DataFrame, DataFrame]]
    convergence = None
    if isinstance(run_results, list):
        results, convergence = run_results[0]
    elif isinstance(run_results, tuple):
        results, convergence = run_results
    else:
        results = run_results

    # -- save Pareto front as CSV ------------------------------------------
    csv_path = os.path.join(seed_dir, f"pareto_front_{seed}.csv")
    results.to_csv(csv_path, index=False)

    # -- save convergence metrics (EpsilonProgress, OperatorProbabilities) --
    if convergence is not None:
        conv_path = os.path.join(seed_dir, f"convergence_{seed}.csv")
        convergence.to_csv(conv_path)
        print(f"    convergence metrics  →  {conv_path}")

    print(f"    {len(results)} Pareto solutions  →  {csv_path}")
    print(f"    convergence archive  →  {os.path.join(seed_dir, archive_filename)}")
    return results


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_optimization_local.py",
        description="Local MOEA optimisation for JUSTICE (Assignment 5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nfe",         type=int,        default=50_000,
                   help="Function evaluations per seed")
    p.add_argument("--seeds",       type=int, nargs="+",
                   default=[9845531, 1644652, 3569126, 6075612, 521475],
                   help="Random seeds")
    p.add_argument("--output_dir",  type=str,
                   default=os.path.join(_SCRIPT_DIR, "results"),
                   help="Root output directory for results (default: assignments_ema/results/)")
    p.add_argument("--n_processes", type=int,        default=None,
                   help="Worker processes (None = cpu_count - 1)")
    p.add_argument("--n_ensembles", type=int,        default=10,
                   help="FAIR ensemble members to use")
    p.add_argument("--config",      type=str,
                   default=os.path.join(_CONFIG_DIR, "config_student.json"),
                   help="Config file path (default: config/config_student.json at repo root)")
    p.add_argument("--population",  type=int,        default=100,
                   help="MOEA population size")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    ema_logging.log_to_stderr(ema_logging.INFO)

    if not os.path.isfile(args.config):
        raise FileNotFoundError(
            f"Config not found: {args.config}\n"
            f"Expected location: {_CONFIG_DIR}/"
        )

    output_dir_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    print("=" * 65)
    print("JUSTICE — Local MOEA Optimisation  (Assignment 5)")
    print("=" * 65)
    print(f"  Welfare function  : UTILITARIAN")
    print(f"  Config            : {args.config}")
    print(f"  NFE per seed      : {args.nfe:,}")
    print(f"  Seeds ({len(args.seeds)})        : {args.seeds}")
    print(f"  FAIR ensembles    : {args.n_ensembles}  "
          f"(indices ≈ {list(np.linspace(1, 1000, args.n_ensembles, dtype=int))})")
    print(f"  Population size   : {args.population}")
    print(f"  Worker processes  : {args.n_processes if args.n_processes else 'auto'}")
    print(f"  Output directory  : {output_dir_abs}")
    print("=" * 65)

    wall_start = time.time()

    for idx, seed in enumerate(args.seeds, start=1):
        print(f"\n[{idx}/{len(args.seeds)}] seed = {seed}")
        print("-" * 40)

        random.seed(seed)
        np.random.seed(seed)

        t0 = time.time()
        run_seed(
            config_path=args.config,
            seed=seed,
            nfe=args.nfe,
            output_dir=output_dir_abs,
            n_ensembles=args.n_ensembles,
            population_size=args.population,
            n_processes=args.n_processes,
        )
        print(f"    elapsed: {(time.time() - t0) / 60:.1f} min")

    total_min = (time.time() - wall_start) / 60
    print(f"\nAll {len(args.seeds)} seeds done in {total_min:.0f} min "
          f"({total_min / 60:.2f} h).")
    print(f"Results → {output_dir_abs}")


if __name__ == "__main__":
    main()

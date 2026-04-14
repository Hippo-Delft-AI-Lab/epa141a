# 📝 Assignments

There are **eight assignments** in EPA141A. They are **not individually graded** but build the analytical foundation for the group project — not completing them seriously will result in a poor project grade. They form a **single end-to-end MBDM pipeline** applied to the **JUSTICE** climate–economy model. Each assignment builds on outputs saved by the previous one — work through them in order.

```{figure} ../_static/JUSTICE_Viz.png
:alt: JUSTICE model architecture — Economy, Climate, Damage, Abatement, and Social Welfare Function components
:width: 100%

JUSTICE couples an economic model (RICE-50+), a climate model (FaIR), damage and abatement functions, and four distributive-justice welfare principles into a single integrated assessment model. [Full model description →](../intro.md)
```

```{warning}
**Do not skip assignments.** Later assignments (A7, A8) load files produced in A6. A6 depends
on A5, which depends on A4. Skipping any assignment breaks the pipeline.
```

## Assignment pipeline

Each assignment reads files produced by earlier ones. The table shows exactly what gets saved and where.

| Assignment | Reads | Writes to `results/` |
|---|---|---|
| A1 — Exploratory Modelling | *(nothing — first run)* | Ensemble experiment results |
| A2 — Sensitivity Analysis | A1 ensemble | Sensitivity indices |
| A3 — Scenario Discovery | A1 ensemble | Scenario boxes |
| A4 — Problem Formulation | *(nothing)* | `config_student.json` |
| A5 — Many-Objective Optimisation | `config_student.json` | Pareto CSVs (one per seed) |
| A6 — MOEA Convergence | A5 Pareto CSVs | `reference_set.csv` |
| A7 — Pareto Visualisation | `reference_set.csv` | Plots and regional maps |
| A8 — Robustness Analysis | `reference_set.csv` | Robustness metrics, re-evaluation cache |

## JUSTICE repository layout

After setup your working directory looks like this. Everything you need for the assignments lives inside these two folders.

```text
epa141a/                              ← open this folder in VS Code / JupyterLab
│
├── assignments_ema/                  ← your assignment notebooks
│   ├── assignment_01_exploratory_modeling.ipynb
│   ├── assignment_02_sensitivity_analysis.ipynb
│   ├── assignment_03_scenario_discovery.ipynb
│   ├── assignment_04_problem_formulation.ipynb
│   ├── assignment_05_moea_local.ipynb
│   ├── assignment_06_moea_convergence.ipynb
│   ├── assignment_07_pareto_visualisation.ipynb
│   ├── assignment_08_robustness.ipynb
│   └── results/                      ← all outputs saved here (CSVs, pickles)
│
└── JUSTICE-main/                     ← the model (cloned separately)
    │
    ├── justice/                      ← Python package — this is what you import
    │   ├── model.py                  ← JUSTICE class — the entry point
    │   ├── welfare/
    │   │   └── social_welfare_function.py   ← 8 distributive justice principles
    │   ├── objectives/
    │   │   └── objective_functions.py       ← years_above_temperature_threshold etc.
    │   ├── util/
    │   │   ├── enumerations.py       ← WelfareFunction enum (pick your principle)
    │   │   ├── EMA_model_wrapper.py  ← bridges JUSTICE ↔ EMA Workbench
    │   │   └── data_loader.py        ← loads SSP/RICE-50 input data
    │   ├── economy/                  ← Cobb-Douglas production (RICE-50+)
    │   ├── emissions/                ← GDP × carbon intensity × (1 − μ)
    │   ├── climate/                  ← FaIR v2.1.3 coupled climate model
    │   ├── damage/                   ← Kalkuhl (2019) damage function
    │   └── abatement/                ← Enerdata MAC abatement cost curves
    │
    ├── config/
    │   ├── JUSTICE_config.yaml       ← model settings (timestep, regions, epsilons)
    │   └── default_parameters.py     ← default uncertainty parameter values
    │
    └── data/                         ← SSP scenarios, RICE-50 regional input data
```

**Key imports used across all assignments:**

| What you import | Where it lives | Used in |
|----------------|----------------|---------|
| `from justice.model import JUSTICE` | `justice/model.py` | A1 – A8 |
| `from justice.util.enumerations import WelfareFunction` | `justice/util/enumerations.py` | A1 – A8 |
| `from justice.objectives.objective_functions import years_above_temperature_threshold` | `justice/objectives/objective_functions.py` | A1 – A3 |
| `from justice.util.EMA_model_wrapper import ...` | `justice/util/EMA_model_wrapper.py` | A4 – A8 |

The setup cell at the top of every notebook calls `os.chdir()` to set the working directory to `JUSTICE-main/` — this is required so the model can find its data files. **Always run the setup cell first.**

---

## Assignment list

| # | Title | Key methods |
|---|-------|-------------|
| [1](assignment_01.md) | Exploratory Modeling with JUSTICE | LHS, EMA Workbench, ensemble runs |
| [2](assignment_02.md) | Sensitivity Analysis: Which Inputs Matter? | Sobol S₁/Sₜ, SALib, Extra-Trees |
| [3](assignment_03.md) | Scenario Discovery: When Do Policies Fail? | PRIM, dimensional stacking |
| [4](assignment_04.md) | Problem Formulation | XLRM, RBF policy representation, EMODPS |
| [5](assignment_05.md) | Many-Objective Optimisation (Local Run) | Borg MOEA, Pareto front |
| [6](assignment_06.md) | MOEA Convergence | ε-dominance, hypervolume, reference set |
| [7](assignment_07.md) | Pareto Visualisation | Parallel coordinates, GeoPandas, RICE-50 maps |
| [8](assignment_08.md) | Robustness Analysis | Satisficing, regret, FaIR ensemble re-evaluation |

## JUSTICE deep uncertain parameters (Assignments 1–3)

In the first three assignments the policy is **fixed** and you explore how model uncertainty affects outcomes.

| Parameter | Description | Range |
|-----------|-------------|-------|
| `ECS` | Equilibrium Climate Sensitivity (°C per CO₂ doubling) | 2.0 – 5.0 |
| `damage_scale` (δ) | Scales the Kalkuhl damage function globally | 0.5 – 2.0 |
| `scenario` | SSP-RCP pair (socioeconomic + emissions pathway) | 8 combinations |
| `welfare_function` | Distributive justice principle for the scalar objective | 8 options |
| `discount_rate` (ρ) | Pure time preference rate | 0.001 – 0.05 |
| `utility_elasticity` (η) | Diminishing marginal utility of consumption | 0.5 – 3.0 |

From Assignment 4 onward the **emission control rate** itself becomes the decision lever, optimised across the 57 RICE-50 regions by a many-objective evolutionary algorithm.

## JUSTICE model outcomes

| Outcome | Description | Goal |
|---------|-------------|------|
| `welfare` | Scalar social welfare aggregated across regions (depends on welfare function) | Minimise |
| `fraction_above_threshold` | Fraction of FaIR ensemble members where global temperature exceeds 2 °C in 2100 | Minimise |
| `welfare_loss_damage` | Welfare loss attributable to climate damages | Maximise |
| `welfare_loss_abatement` | Welfare loss attributable to abatement costs | Maximise |

```{note}
The same emission policy can rank **very differently** depending on which welfare function is
applied. This normative uncertainty is a central theme of the JUSTICE assignments.
```

---

## Troubleshooting

:::{dropdown} `FileNotFoundError: reference_set_utilitarian.csv not found` (A7 or A8)
Assignment 7 and 8 both load the reference set produced by Assignment 6. If the file is missing:

1. Open `assignment_06_convergence.ipynb` and run all cells in order.
2. Confirm that `results/reference_set.csv` (or `reference_set_utilitarian.csv`) now exists.
3. Re-open A7 or A8 and re-run.

This file is not provided — you must generate it yourself in A6.
:::

:::{dropdown} Assignment 8 first run can take a long time
A8 re-evaluates policies across a large ensemble of uncertain futures. This is slow on the first run. Results are cached and loaded instantly on subsequent runs.

If you want to run it faster while testing, reduce `n_scenarios` in the notebook before running the full re-evaluation.
:::

:::{dropdown} `RecursionError` during `perform_experiments`
This is a known incompatibility between Python 3.14 and `tqdm ≥ 4.67` inside Jupyter. Add this line at the top of the cell and restart the kernel:

```python
import tqdm; tqdm.tqdm = tqdm.std.tqdm
```

Then re-run the cell.
:::

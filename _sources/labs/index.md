# 🔬 Labs

There are **six lab sessions** in EPA141A. Labs use the **Lake Problem** — a stylised, fast-running model — to let you experiment freely with MBDM methods before applying them to the more complex JUSTICE model in the assignments.

Labs are **not graded** but are essential preparation for the assignments.

## Lab list

| # | Title | Key methods |
|---|-------|-------------|
| [1](lab_01.md) | Exploratory Modelling | EMA Workbench, Monte Carlo, pairs scatter |
| [2](lab_02.md) | Global Sensitivity Analysis | Sobol S₁/Sₜ, SALib, Extra-Trees, LHS |
| [3](lab_03.md) | Scenario Discovery: PRIM and Dimensional Stacking | PRIM, scenario boxes |
| [4](lab_04.md) | MOEA Setup and Optimisation | ε-NSGA-II, Pareto dominance, ε-progress |
| [5](lab_05.md) | Pareto Analysis, Visualisation, and Convergence | Hypervolume, GD, ε-indicator, seed analysis |
| [6](lab_06.md) | Multi-Objective Robust Decision Making (MORDM) | Signal-to-noise, max regret, DPS/RBF, PRIM |

## The Lake Problem

The **shallow lake eutrophication problem** is the classic teaching model used in Labs 1–5. It is small enough to run thousands of simulations in seconds, making it ideal for exploring methods.

A municipality releases phosphorus into a shallow lake year by year. If pollution exceeds a critical threshold, the lake flips into an **irreversible eutrophic state**. The question is: *how much phosphorus can we safely release each year while maximising economic benefit?*

The lake model comes in two versions:

- **Intertemporal** (Labs 1–5) — the policy is a fixed schedule of 100 annual release rates. Lever space: 100-dimensional.
- **DPS / Direct Policy Search** (Lab 6) — the policy is a *closed-loop rule* that adapts to the current phosphorus level via five Radial Basis Functions. Lever space: 5-dimensional. This produces more realistic, adaptive policies.

### Uncertain parameters

| Symbol | Name | Range |
|--------|------|-------|
| `b` | Phosphorus removal rate | — |
| `q` | Recycling nonlinearity exponent | — |
| `mean` | Mean natural inflow | — |
| `stdev` | Variability of natural inflow | — |
| `delta` | Discount rate (time preference) | — |

### Outcomes

| Outcome | What it measures | Optimisation goal |
|---------|-----------------|-------------------|
| `max_P` | Peak phosphorus concentration | Minimise |
| `utility` | Discounted sum of utility from releases | Maximise |
| `inertia` | Fraction of years with small year-to-year change | Maximise |
| `reliability` | Fraction of years the lake stays below the critical threshold | Maximise |

```{note}
The lake model is already included in the `ema_workbench` package — no separate installation
needed. The files `labs/lakemodel_function.py` and `labs/dps_lake_model.py` are thin wrappers
that make it compatible with the EMA Workbench API.
```

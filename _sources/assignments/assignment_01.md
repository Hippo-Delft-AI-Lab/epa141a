# Assignment 1 — Exploratory Modeling with JUSTICE

**Key methods:** LHS · EMA Workbench · Ensemble runs · Outcome distributions

---

## Learning objectives

After completing this assignment you will be able to:

1. Verify that the JUSTICE model is installed and runs correctly.
2. Identify and define **model input uncertainties** — both physical (ECS ensemble) and normative (ρ, η, δ) parameters.
3. Use **Latin Hypercube Sampling** to generate a scenario set that spans the uncertainty space.
4. Run an exploratory ensemble with `ema_workbench` and save results.
5. Visualise and interpret outcome distributions in terms of model behaviour under deep uncertainty.

## JUSTICE uncertainties in this assignment

| Parameter | Symbol | Type | Range |
|-----------|--------|------|-------|
| Pure rate of time preference | ρ (rho) | Normative | [0.001, 0.030] |
| Elasticity of marginal utility | η (eta) | Normative | [0.5, 1.5] |
| Damage function scale factor | δ (delta) | Normative | [0.5, 2.0] |
| FaIR ensemble member index | `ecs_ensemble` | Physical | [1, 1001] |

## Outputs saved for later assignments

The ensemble results are saved to `results/` and loaded by Assignments 2 and 3.

## Notebook

`assignments_ema/assignment_01_exploratory_modeling.ipynb`

## Prerequisites

- `epa141a` environment installed and `EPA141A (JUSTICE)` kernel selected.
- Run **Exercise 0** first to verify imports.

```{note}
A single JUSTICE run (2015–2300 at 1-year timestep) takes roughly 1–3 seconds.
The 100-scenario ensemble in this assignment takes 5–15 minutes sequentially.
```

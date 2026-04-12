# Lab 2 — Global Sensitivity Analysis

**Key methods:** Sobol S₁/Sₜ · SALib · Extra-Trees feature scoring · LHS

---

## Background

**Global Sensitivity Analysis (GSA)** quantifies how much each uncertain input parameter contributes to the variance of model outputs *across the entire uncertainty space*. This is different from local sensitivity analysis, which perturbs one parameter at a time near a nominal value.

This lab contrasts three complementary GSA methods:

| Method | Approach | Advantage |
|--------|----------|-----------|
| OLS regression | Linear correlation coefficients | Fast, interpretable |
| **Sobol S₁ and Sₜ** | Variance decomposition (SALib) | Captures non-linear effects and interactions |
| **Extra-Trees feature scoring** | Random forest importance (EMA Workbench) | Model-free, handles arbitrary response surfaces |

## What you will do

1. Run a **Saltelli-sampled** ensemble of the Lake Problem.
2. Compute **Sobol first-order (S₁)** and **total-order (Sₜ)** indices with SALib.
3. Apply `ema_workbench` **Extra-Trees feature scoring**.
4. Compare the three rankings and discuss where they agree and diverge.
5. Interpret the S₁/Sₜ gap in terms of parameter interactions.

## Notebook

`labs/lab_02_sensitivity_analysis.ipynb`

## What this prepares you for

→ **Assignment 2** — sensitivity analysis on JUSTICE outputs (welfare, temperature, damages).

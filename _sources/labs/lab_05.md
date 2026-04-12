# Lab 5 — Pareto Analysis, Visualisation, and Convergence

**Key methods:** Hypervolume · Generational Distance (GD) · ε-indicator · Seed analysis · Parallel coordinates

---

## Background

Having found a Pareto-approximate set, two questions remain:

1. **Has the algorithm converged?** Are there still unexplored regions of objective space?
2. **How much does the result depend on the random seed?** MOEAs are stochastic — different seeds may produce different fronts.

Six convergence metrics are used in this lab:

| Metric | What it measures |
|--------|-----------------|
| **Hypervolume** | Volume of objective space dominated by the current set (higher = better) |
| **Generational Distance (GD)** | Average distance from current set to reference set (lower = better) |
| **Inverted GD** | Average distance from reference set to current set |
| **ε-indicator** | Smallest ε such that current set ε-dominates reference set |
| **Spacing** | Uniformity of solutions along the front |
| **ε-progress** | Whether new non-dominated solutions were added in the last NFE block |

## What you will do

1. Load archived Pareto sets from **multiple MOEA seeds** (pre-computed for the Lake Problem).
2. Compute all six convergence metrics and plot them as a function of NFE.
3. Build a **cross-seed reference set** using epsilon-nondominated sorting.
4. Visualise the final Pareto front with **parallel coordinates** (Plotly).

## Notebook

`labs/lab_05_pareto_analysis_convergence.ipynb`

## What this prepares you for

→ **Assignment 6** — convergence analysis and reference set construction for the JUSTICE Pareto results.
→ **Assignment 7** — parallel coordinates and world-map visualisation of the JUSTICE reference set.

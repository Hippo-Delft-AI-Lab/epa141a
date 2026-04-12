# Lab 4 — MOEA Setup and Optimisation

**Key methods:** ε-NSGA-II · Platypus · Pareto dominance · ε-progress

---

## Background

In the previous labs, you *sampled* the lever space to find candidate policies. In this lab you **search** it using a **Many-Objective Evolutionary Algorithm (MOEA)**, which iteratively evolves a population of solutions toward the Pareto-optimal front.

**ε-NSGA-II** is the algorithm used throughout this course. The ε parameters control the resolution of the Pareto front — larger ε values produce a coarser but more stable approximation.

## What you will do

1. Define the Lake Problem as a multi-objective optimisation with `ema_workbench`.
2. Configure **ε-NSGA-II** in Platypus: set population size, number of function evaluations, and ε values for each objective.
3. Run the MOEA and collect the **Pareto-approximate set**.
4. Plot the resulting Pareto front in objective space.
5. Experiment: how do different ε values and NFE budgets affect the front?

## Notebook

`labs/lab_04_moea_setup_and_optimization.ipynb`

## What this prepares you for

→ **Assignment 5** — running ε-NSGA-II on the JUSTICE model (a much larger problem with 57-dimensional lever space and multiple competing welfare objectives).

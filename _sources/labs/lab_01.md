# Lab 1 — Exploratory Modelling with the Lake Problem

**Key methods:** EMA Workbench · Monte Carlo · Latin Hypercube Sampling · Pairs scatter

---

## Background

The **Lake Problem** is a stylised decision problem in which the inhabitants of a lakeside town decide how much phosphorus to release into a lake each year. If the pollution in the lake exceeds a critical threshold, the lake undergoes irreversible eutrophication. The dynamics are governed by:

$$P_{t+1} = P_t + a_t + \frac{P_t^q}{1 + P_t^q} - bP_t + \epsilon_t$$

where $P_t$ is the phosphorus level, $a_t$ is the anthropogenic release, $b$ is the natural removal rate, $q$ controls the recycling nonlinearity, and $\epsilon_t$ is stochastic natural inflow.

## What you will do

1. Connect the lake model to **EMA Workbench** as a `Model` object.
2. Specify uncertain parameters (lake physics) and lever policies (release schedules).
3. Run **open exploration** across uncertain parameters and multiple policies with Latin Hypercube Sampling.
4. Visualise outcome distributions and pairwise trade-offs (pairs scatter matrix).
5. Interpret the outcome space — which policy performs well across many futures?

## Notebook

`labs/lab_01_exploratory_modelling.ipynb`

## What this prepares you for

→ **Assignment 1** — the same EMA workflow applied to the JUSTICE model.

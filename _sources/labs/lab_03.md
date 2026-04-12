# Lab 3 — Scenario Discovery: PRIM and Dimensional Stacking

**Key methods:** PRIM · Patient Rule Induction Method · Dimensional stacking · Scenario boxes

---

## Background

**Scenario discovery** asks: *under what combinations of uncertain conditions does a policy fail?* Rather than asking *what is the optimal policy?*, it characterises the region of the uncertainty space where an outcome becomes unacceptable.

**PRIM (Patient Rule Induction Method)** finds the smallest hyper-box in the uncertainty space that contains a disproportionate share of the "failure" cases. Two key statistics describe each box:

- **Coverage**: fraction of all failures captured by the box.
- **Density**: fraction of cases inside the box that are failures.

There is a coverage–density trade-off, visualised as the **peeling trajectory**.

## What you will do

1. Fix a policy and define a binary outcome (lake fails / does not fail).
2. Apply **PRIM** to find the uncertainty box that best explains failures.
3. Inspect the peeling trajectory and choose a box that balances coverage and density.
4. Describe the scenario box in plain language: *"The policy fails when X is high AND Y is low."*
5. Visualise failure conditions with **dimensional stacking**.

## Notebook

`labs/lab_03_scenario_discovery.ipynb`

## What this prepares you for

→ **Assignment 3** — scenario discovery on JUSTICE outcomes (when does a fixed emission policy lead to unacceptable welfare outcomes?).

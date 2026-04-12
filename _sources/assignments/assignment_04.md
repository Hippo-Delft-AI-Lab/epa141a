# Assignment 4 — Problem Formulation

**Key methods:** XLRM framework · RBF policy representation · EMODPS

---

## Learning objectives

1. Complete the **XLRM framework** for the JUSTICE climate policy problem under deep uncertainty.
2. Justify your choice of **objectives (X)**, **levers (L)**, **relationships (R)**, and **exogenous uncertainties (M)**.
3. Choose a **policy representation**: the Emission Control Rate (ECR) will be encoded as an adaptive **Radial Basis Function (RBF)** policy (EMODPS).
4. Translate the XLRM framework into a formal `ema_workbench` `Model` specification.
5. Save the problem formulation as `config_student.json` for use in Assignments 5–8.

## JUSTICE policy representation

In Assignments 4–8, the policy is no longer a fixed emission level — it is a **closed-loop adaptive rule** that adjusts the emission control rate (ECR) in response to the current global temperature. This is encoded as **Radial Basis Functions (RBF)**, a flexible and smooth function approximation.

The levers are the **parameters of the RBF** — centres, radii, and weights — rather than the emission rates directly.

## Output

`config_student.json` — saved to `config/` and loaded by Assignments 5, 6, 7, 8.

## Notebook

`assignments_ema/assignment_04_problem_formulation.ipynb`

## Submission

Submit `A4_<studentnumber>.ipynb` via Brightspace with all cells executed **and `config_student.json` saved**.

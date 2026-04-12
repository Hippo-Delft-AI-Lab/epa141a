# Lab 6 — Multi-Objective Robust Decision Making (MORDM)

**Key methods:** MORDM · Signal-to-noise robustness · Max regret · DPS/RBF policies · PRIM scenario re-discovery

---

## Background

**MORDM** (Kasprzyk et al., 2013) is a structured methodology for finding and evaluating robust policies under deep uncertainty. It consists of four iterative steps:

| Step | Name | Question answered |
|------|------|-------------------|
| 1 | Formulation | What are the objectives, levers, and uncertainties? |
| 2 | Search | What Pareto-approximate policies exist? |
| 3 | Re-evaluation | How do policies perform across a large ensemble of futures? |
| 4 | Robustness analysis | Which policies are least regret / satisficing? |

This lab applies the **full MORDM cycle** to the **DPS (Direct Policy Search) version** of the Lake Problem, in which the policy is a closed-loop adaptive rule encoded as five Radial Basis Functions. This 5-dimensional lever space is much more tractable than the 100-dimensional intertemporal version.

## What you will do

1. **Optimise** the DPS lake model with ε-NSGA-II to find a Pareto-approximate set.
2. **Re-evaluate** the Pareto set across 1,000 sampled uncertain futures.
3. **Compute robustness metrics**: signal-to-noise ratio and maximum regret across the four lake objectives.
4. **Rank** policies by robustness and select a shortlist.
5. **Scenario re-discovery**: apply PRIM to find the uncertain conditions under which even the most robust policies still fail.

## Notebook

`labs/lab_06_mordm.ipynb`

## What this prepares you for

→ **Assignment 8** — robustness analysis (satisficing and regret) on the JUSTICE Pareto policies, including scenario re-discovery using PRIM.

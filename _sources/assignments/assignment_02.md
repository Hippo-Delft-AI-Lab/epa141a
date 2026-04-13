# Assignment 2 — Sensitivity Analysis: Which Inputs Matter?

**Key methods:** Sobol S₁/Sₜ · SALib · Extra-Trees feature scoring

---

## Learning objectives

1. Understand the purpose and limitations of **variance-based sensitivity analysis**.
2. Compute **Sobol first-order (S₁) and total-order (Sₜ)** sensitivity indices using SALib.
3. Use **Extra-Trees feature scoring** as a complementary, model-free approach.
4. Rank the four uncertain inputs by their contribution to each JUSTICE outcome.
5. Critically compare Sobol indices and feature scoring results: where do they agree? Where do they diverge and why?

## JUSTICE outcomes analysed

| Outcome | What it measures |
|---------|-----------------|
| `welfare` | Aggregate discounted social welfare (all regions, all years) |
| `years_above_temperature_threshold` | Years global mean temperature exceeds 2 °C |
| `welfare_loss_damage` | Welfare lost due to climate damages |
| `welfare_loss_abatement` | Welfare lost due to emission reduction costs |

## Prerequisites

- **Assignment 1 completed and results saved** — this assignment reads the A1 ensemble.

## Notebook

`assignments_ema/assignment_02_sensitivity_analysis.ipynb`

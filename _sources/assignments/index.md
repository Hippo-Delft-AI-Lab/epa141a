# 📝 Assignments

There are **eight assignments** in EPA141A. They are **not individually graded** but build the analytical foundation for the group project — not completing them seriously will result in a poor project grade. They form a **single end-to-end MBDM pipeline** applied to the **JUSTICE** climate–economy model. Each assignment builds on outputs saved by the previous one — work through them in order.

```{warning}
**Do not skip assignments.** Later assignments (A7, A8) load files produced in A6. A6 depends
on A5, which depends on A4. Skipping any assignment breaks the pipeline.
```

## Assignment pipeline

```
A1 → ensemble runs saved to results/
A2 → reads A1 ensemble → sensitivity indices
A3 → reads A1 ensemble → scenario boxes
A4 → defines objectives & policy structure → config_student.json
A5 → runs MOEA → Pareto CSVs in results/
A6 → merges A5 seeds → reference_set.csv
A7 → reads A6 reference set → plots & maps
A8 → reads A6 reference set → robustness metrics
```

## Assignment list

| # | Title | Key methods |
|---|-------|-------------|
| [1](assignment_01.md) | Exploratory Modeling with JUSTICE | LHS, EMA Workbench, ensemble runs |
| [2](assignment_02.md) | Sensitivity Analysis: Which Inputs Matter? | Sobol S₁/Sₜ, SALib, Extra-Trees |
| [3](assignment_03.md) | Scenario Discovery: When Do Policies Fail? | PRIM, dimensional stacking |
| [4](assignment_04.md) | Problem Formulation | XLRM, RBF policy representation, EMODPS |
| [5](assignment_05.md) | Many-Objective Optimisation (Local Run) | Borg MOEA, Pareto front |
| [6](assignment_06.md) | MOEA Convergence | ε-dominance, hypervolume, reference set |
| [7](assignment_07.md) | Pareto Visualisation | Parallel coordinates, GeoPandas, RICE-50 maps |
| [8](assignment_08.md) | Robustness Analysis | Satisficing, regret, FaIR ensemble re-evaluation |

## JUSTICE deep uncertain parameters (Assignments 1–3)

In the first three assignments the policy is **fixed** and you explore how model uncertainty affects outcomes.

| Parameter | Description | Range |
|-----------|-------------|-------|
| `ECS` | Equilibrium Climate Sensitivity (°C per CO₂ doubling) | 2.0 – 5.0 |
| `damage_scale` (δ) | Scales the Kalkuhl damage function globally | 0.5 – 2.0 |
| `scenario` | SSP-RCP pair (socioeconomic + emissions pathway) | 8 combinations |
| `welfare_function` | Distributive justice principle for the scalar objective | 8 options |
| `discount_rate` (ρ) | Pure time preference rate | 0.001 – 0.05 |
| `utility_elasticity` (η) | Diminishing marginal utility of consumption | 0.5 – 3.0 |

From Assignment 4 onward the **emission control rate** itself becomes the decision lever, optimised across the 57 RICE-50 regions by a many-objective evolutionary algorithm.

## JUSTICE model outcomes

| Outcome | Description | Goal |
|---------|-------------|------|
| `welfare` | Scalar social welfare aggregated across regions (depends on welfare function) | Minimise |
| `fraction_above_threshold` | Fraction of FaIR ensemble members where global temperature exceeds 2 °C in 2100 | Minimise |
| `welfare_loss_damage` | Welfare loss attributable to climate damages | Maximise |
| `welfare_loss_abatement` | Welfare loss attributable to abatement costs | Maximise |

```{note}
The same emission policy can rank **very differently** depending on which welfare function is
applied. This normative uncertainty is a central theme of the JUSTICE assignments.
```

## Submission guidelines

- Submit via **Brightspace** as a `.ipynb` file with all cells executed.
- Include your **student number** in the filename: `A1_5xxxxxx.ipynb`.
- All code must be reproducible: restart kernel → run all cells → no errors.
- Select the **`EPA141A (JUSTICE)`** kernel before submitting.
- Late penalty: **−10 % per 24 h** unless an extension is agreed in advance.

# EPA141A — Model-Based Decision Making

**MSc Engineering and Policy Analysis · Delft University of Technology**

---

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🚀 New student? Start here
:link: start-here
:link-type: doc

Set up your environment and run your first notebook in under 30 minutes.
+++
[→ Get started](start-here.md)
:::

:::{grid-item-card} 📅 What's happening this week
:link: schedule
:link-type: doc

Check the lecture topic, lab session, and assignment deadline for this week.
+++
[→ View schedule](schedule.md)
:::

::::

---

## What this course is about

This course teaches you how to use **computational models to support decisions on complex societal problems** — climate change, energy transition, water management — where the future is deeply uncertain and stakeholders disagree on values.

You will work with a real climate Integrated Assessment Model (**JUSTICE**), run thousands of simulations, explore what happens under different scenarios and assumptions, design policies that are robust to uncertainty, and then defend your recommendations in a structured debate simulating **climate change negotiations**.

By the end of the course you will be able to run a full model-based policy analysis pipeline: from exploratory modelling and sensitivity analysis, through multi-objective optimisation, to robustness analysis and stakeholder communication.

---

## Course at a glance

| What | Where |
|------|-------|
| 8 individual assignments (JUSTICE model) | [Assignments](assignments/index.md) |
| 6 guided lab sessions (Lake Problem) | [Labs](labs/index.md) |
| 13 lectures | [Lectures](lectures.md) |
| Climate change debate (group project) | [Debate](debate/index.md) |
| Final project report | [Final Project](project/index.md) |
| Python environment setup | [Installation](installation/index.md) |
| Weekly schedule and deadlines | [Schedule](schedule.md) |

**Assessment:** entirely through a group project — no written exam. Groups of 4 students each take a climate actor role and develop a global mitigation strategy using JUSTICE.

| Component | Weight |
|-----------|--------|
| Part 1: Model-based analysis and policy advice | 70 % |
| Part 2: Political reflection on the climate change debate | 30 % |

---

## The JUSTICE model

**JUSTICE** (JUST Integrated Climate Economy) is an open-source Integrated Assessment Model developed at TU Delft's HIPPO Lab. It simulates the global economy, the carbon cycle, and the climate system as a coupled dynamic model running from 2015 to 2300 across **57 world regions**.

```{figure} _static/justice_regions_map.png
:alt: JUSTICE 57 simulated world regions
:width: 100%

JUSTICE simulates 57 world regions based on the RICE-50+ framework.
```

The model converts a climate policy into welfare outcomes through seven linked steps:

1. **Economy** — Cobb-Douglas production across 57 RICE-50 regions
2. **Emissions** — GDP × carbon intensity × (1 − μ), where μ is the emission control rate
3. **Cumulative CO₂** — feeds into FaIR v2.1.3 → global mean temperature
4. **Damages** — Kalkuhl (2019) function converts temperature to % GDP loss
5. **Abatement costs** — Enerdata MAC curves, rising non-linearly with μ
6. **Net consumption** — damages and abatement subtracted from gross output
7. **Social welfare function** — one of eight distributive justice principles aggregates into a scalar objective

The key insight: **the same emission policy can rank very differently depending on which welfare function is applied.** This makes normative uncertainty explicit and quantitative.

---

## Learning objectives

After completing this course, students can:

1. **Explain** the theoretical foundations of science-based and deliberative policy analysis, including contested knowledge claims, decision-making under deep uncertainty, and adaptive policy-making.
2. **Apply** exploratory modelling and scenario discovery techniques to a policy case, translating model results into evidence-based recommendations for decision-makers.
3. **Analyse** how model-based decision support interacts with multi-actor political processes, identifying how normative assumptions and political context shape analytical choices and their outcomes.
4. **Co-create** a model-based policy analysis project in a team setting, synthesising theoretical frameworks and model results to construct and defend a stakeholder position in a structured debate.
5. **Reflect** critically on the effects, limitations, and risks of model-based policy advice in context, relating these to the ultimate goal of producing more effective and legitimate public policies.

---

## Staff

| Role | Name | Email |
|------|------|-------|
| Module manager & core lecturer | Jazmin Zatarain Salazar | [J.ZatarainSalazar@tudelft.nl](mailto:J.ZatarainSalazar@tudelft.nl) |
| Lecturer | Jan Kwakkel | [j.h.kwakkel@tudelft.nl](mailto:j.h.kwakkel@tudelft.nl) |
| Lecturer | Tamara Metze | [T.A.P.Metze@tudelft.nl](mailto:T.A.P.Metze@tudelft.nl) |
| Lecturer | Patrick Steinmann | [P.Steinmann@tudelft.nl](mailto:P.Steinmann@tudelft.nl) |
| Teaching assistant | Rachel Delvin Sutiono | [RachelDelvinSutiono@student.tudelft.nl](mailto:RachelDelvinSutiono@student.tudelft.nl) |
| Teaching assistant | Move Phutthaphaiboon | [T.Phutthaphaiboon@student.tudelft.nl](mailto:T.Phutthaphaiboon@student.tudelft.nl) |

**Lectures:** Monday & Wednesday · 13:15–15:00
**Lab sessions:** Monday & Wednesday · 15:00–17:00

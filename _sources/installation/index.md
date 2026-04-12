# 🐍 Software Installation

To run the EPA141A assignments and labs you need two repositories and a working **Python 3.12 environment** with the course packages installed. We use **conda** to manage the environment.

## Course repositories

Everything you need is hosted under the **[Hippo-Delft-AI-Lab](https://github.com/Hippo-Delft-AI-Lab)** organisation on GitHub.

| Repository | What it contains | Link |
|------------|-----------------|------|
| **epa141a** | Assignment notebooks, lab notebooks, `environment.yml`, configuration files | [github.com/Hippo-Delft-AI-Lab/epa141a](https://github.com/Hippo-Delft-AI-Lab/epa141a) |
| **JUSTICE** | The climate Integrated Assessment Model used in all assignments and labs | [github.com/Hippo-Delft-AI-Lab/JUSTICE](https://github.com/Hippo-Delft-AI-Lab/JUSTICE) |

```{admonition} How the two repositories fit together
:class: note
JUSTICE must be placed **inside** the `epa141a/` folder, in a sub-folder called `JUSTICE-main/`.
The installation guides below walk you through this step-by-step.

```
epa141a/          ← course repository (clone this first)
└── JUSTICE-main/ ← JUSTICE model (clone this inside epa141a/)
└── assignments_ema/
└── labs/
└── environment.yml
\`\`\`
```

```{admonition} Allow 15–30 minutes
:class: tip
The environment contains several large packages (GeoPandas, FaIR, EMA Workbench).
A fresh install typically takes 15–30 minutes depending on your internet speed.
```

```{admonition} Complete setup & orientation guide
:class: seealso

Need everything in one place — models, labs, assignments, and installation — with worked code examples and troubleshooting? Open the **[Setup & Orientation guide](../_static/setup_and_orientation.html)** (standalone HTML, opens in the same tab).
```

## The big picture

If you have never set up a programming environment before, the number of tools can feel overwhelming. Here is how they all fit together — read this before installing anything.

```
┌─────────────────────────────────────────────────────────┐
│                        GitHub                           │
│   Hippo-Delft-AI-Lab/epa141a   +   JUSTICE             │
│   (assignments, labs, config)      (climate model)      │
└──────────────────┬──────────────────────────────────────┘
                   │  git clone  (download once)
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   Your laptop                           │
│                                                         │
│  epa141a/              ← course folder                  │
│  └── JUSTICE-main/     ← climate model (inside it)     │
│  └── assignments_ema/                                   │
│  └── labs/                                              │
│  └── environment.yml   ← package list                   │
└──────────────────┬──────────────────────────────────────┘
                   │  conda env create  (install once)
                   ▼
┌─────────────────────────────────────────────────────────┐
│           conda environment  "epa141a"                  │
│   Python 3.12 + all course packages pre-installed       │
│   (numpy, ema-workbench, platypus, fair, …)             │
└──────────────────┬──────────────────────────────────────┘
                   │  open notebook + select kernel
                   ▼
┌─────────────────────────────────────────────────────────┐
│         VS Code  or  JupyterLab  (your editor)          │
│   Opens .ipynb notebooks → runs code in the             │
│   "epa141a" environment → produces results              │
└─────────────────────────────────────────────────────────┘
```

**In plain English:** GitHub is where the course files live. You download them once with `git clone`. `conda` installs all the Python packages you need into a clean, isolated environment. VS Code or JupyterLab is the editor where you open and run the notebooks — it connects to that environment to execute the code.

---

## What is a terminal?

A **terminal** (also called a command line, shell, or console) is a text-based way to give instructions to your computer. Instead of clicking icons, you type commands and press Enter.

**How to open one:**

::::{tab-set}

:::{tab-item} macOS
Press `⌘ Space`, type **Terminal**, and press Enter.
Alternatively, open VS Code and press `` Ctrl ` `` to open a terminal inside it.
:::

:::{tab-item} Windows
Search for **Anaconda Prompt** in the Start Menu and open it.
Do not use the regular Command Prompt or PowerShell for conda commands — they will not recognise `conda` unless you have configured them separately.
:::

:::{tab-item} Linux
Press `Ctrl Alt T` or search for **Terminal** in your application launcher.
:::

::::

**Three commands you actually need:**

| Command | What it does | Example |
|---------|-------------|---------|
| `pwd` | Print current directory (where am I?) | `/Users/you/courses/epa141a` |
| `ls` (macOS/Linux) or `dir` (Windows) | List files in the current folder | shows `assignments_ema/`, `labs/`, … |
| `cd folder` | Move into a folder | `cd epa141a` |

```{tip}
You do not need to become a terminal expert. The installation guides tell you exactly which commands to run and when.
```

---

## What is a virtual environment?

Think of a virtual environment as a **clean, isolated workspace** for a specific project. Each environment has its own copy of Python and its own set of packages, completely separate from every other environment on your machine.

**Why this matters for you:**

- Different courses and projects often need different package versions. Without environments they conflict with each other and break.
- The `epa141a` environment has `fair==2.1.3` pinned exactly. If you installed JUSTICE into your base Python, another course might accidentally upgrade it and break all your assignments.
- If something goes wrong you can delete and recreate the environment without touching anything else on your laptop.

**The one habit that prevents 90% of errors:** always make sure the `epa141a` environment is **active** (selected as the kernel in VS Code, or shown as `(epa141a)` in your terminal prompt) before running any course code.

---

## Choose your installation method

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🖱️ Graphical Installation
:link: graphical
:link-type: doc

**Recommended for most students.**
Uses Anaconda Navigator — minimal terminal required.
+++
[→ Graphical Installation](graphical.md)
:::

:::{grid-item-card} ⌨️ Command-Line Installation
:link: cli
:link-type: doc

**Faster and more reliable.**
Uses the `conda` terminal command.
+++
[→ Command-Line Installation](cli.md)
:::

:::{grid-item-card} 🆚 VS Code Setup
:link: vscode
:link-type: doc

**Recommended editor.**
Run notebooks directly in VS Code — no browser needed.
+++
[→ VS Code Setup](vscode.md)
:::

::::

## What gets installed

The `epa141a` conda environment includes:

| Package | Version | Used in |
|---------|---------|---------|
| Python | 3.12 | All |
| JupyterLab | ≥ 4.0 | All |
| `ema-workbench` | ≥ 3.0 | All assignments & labs |
| `numpy`, `pandas` | latest stable | All |
| `matplotlib`, `seaborn` | latest stable | All |
| `scipy` | ≥ 1.10 | All |
| `SALib` | ≥ 1.4 | Assignment 2, Lab 2 |
| `platypus-opt` | ≥ 1.1 | Assignments 5–8, Labs 4–6 |
| `deap` | ≥ 1.4 | Assignments 5–8 |
| `ipyparallel` | ≥ 8.6 | Assignments 5–6 |
| `plotly` | ≥ 5.19 | Assignment 7, Lab 5 |
| `geopandas` | ≥ 0.14 | Assignment 7 |
| `fair` | **2.1.3 (exact)** | All (JUSTICE climate model) |
| `h5py` | ≥ 3.10 | All (JUSTICE HDF5 data files) |
| `pycountry` | ≥ 24.6 | All (JUSTICE region mapping) |
| `kaleido` | ≥ 0.2 | Assignment 7 (Plotly figure export) |
| `tqdm` | ≥ 4.0 | Assignments 1, 2, 3, 8 |

```{warning}
`fair` must be version **2.1.3 exactly**. JUSTICE was developed and tested against this version.
Do not upgrade it.
```

## Two-part installation

The setup has two parts:

1. **Create the conda environment** — installs all Python packages.
2. **Install JUSTICE in editable mode** — makes the `justice`, `solvers`, and `analysis` packages importable.

Both parts are covered step-by-step in the installation guides above.

## Verifying the installation

After installation, open `assignments_ema/assignment_01_exploratory_modeling.ipynb` and run **Exercise 0** (the first code cell). You should see output like:

```
✓  justice
✓  numpy
✓  pandas
✓  matplotlib
✓  ema_workbench
✓  scipy
✓  seaborn

Python 3.12.x
```

Every package should show `✓` with no red errors. If anything shows `✗`, see the Troubleshooting section at the bottom of the installation guide you followed.

## Kernel name

The Jupyter kernel is registered as **`EPA141A (JUSTICE)`**. Always select this kernel (not the base Python kernel) when opening assignment or lab notebooks.

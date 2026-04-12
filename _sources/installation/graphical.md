# Graphical Installation (Anaconda Navigator)

This guide walks you through setting up the course environment using **Anaconda Navigator** — the terminal is only needed for two short steps (downloading JUSTICE and registering the kernel).

```{admonition} Allow 15–30 minutes
:class: tip
A fresh install typically takes 15–30 minutes. Keep Anaconda Navigator open and your computer awake throughout.
```

---

## Step 1 — Install Miniconda or Anaconda

If you already have Miniconda or Anaconda installed, skip to Step 2 below.

::::{tab-set}

:::{tab-item} Miniconda (recommended — smaller download)
1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
2. Download the **Python 3.12** installer for your operating system.
3. Run the installer and accept the default settings.
   - **Windows**: tick *"Add Miniconda3 to my PATH"* if prompted, or use **Anaconda Prompt** for conda commands.
   - **macOS/Linux**: the installer modifies your shell profile automatically — open a **new** terminal after installation.

Open **Anaconda Navigator** from the Start Menu (Windows) or Applications (macOS).
:::

:::{tab-item} Anaconda (full distribution)
1. Go to the [Anaconda download page](https://www.anaconda.com/products/distribution).
2. Download the **Python 3.12** installer for your OS (≈ 800 MB).
3. Run the installer and follow the on-screen instructions.

Anaconda Navigator is included — open it from the Start Menu or Launchpad.
:::

::::

---

## Step 2 — Download the course repository

The course repository is at **[github.com/Hippo-Delft-AI-Lab/epa141a](https://github.com/Hippo-Delft-AI-Lab/epa141a)**. It contains all assignment and lab notebooks, configuration files, and the `environment.yml` used in the next step.

::::{tab-set}

:::{tab-item} Download ZIP (no Git required)
1. Go to [github.com/Hippo-Delft-AI-Lab/epa141a](https://github.com/Hippo-Delft-AI-Lab/epa141a).
2. Click **Code → Download ZIP**.
3. Unzip to a location **without spaces** in the path (e.g. `C:\courses\epa141a` or `~/courses/epa141a`).
:::

:::{tab-item} Git clone
Open a terminal and run:
```bash
git clone https://github.com/Hippo-Delft-AI-Lab/epa141a.git
cd epa141a
```
:::

::::

---

## Step 3 — Clone the JUSTICE model

JUSTICE is maintained in a **separate repository** at **[github.com/Hippo-Delft-AI-Lab/JUSTICE](https://github.com/Hippo-Delft-AI-Lab/JUSTICE)** and must be placed **inside** the course repo as a folder called `JUSTICE-main/`.

Open a terminal, navigate to your course repo folder, and run:

```bash
git clone https://github.com/Hippo-Delft-AI-Lab/JUSTICE.git JUSTICE-main
```

```{warning}
The folder **must** be named `JUSTICE-main/` — the assignment notebooks and the install command in Step 5 both expect this exact name.
```

---

## Step 4 — Create the conda environment in Anaconda Navigator

1. Open **Anaconda Navigator**.
2. Click **Environments** in the left sidebar.
3. At the bottom of the environment list, click **Import**.
4. In the dialog:
   - **Name**: type `epa141a`
   - **Specification File**: click the folder icon and select the `environment.yml` file from the course repo root.
5. Click **Import** and wait for the progress bar to finish (15–30 minutes).

```{admonition} Don't close Navigator
:class: warning
Keep Anaconda Navigator open and your computer awake during the installation.
```

---

## Step 5 — Install JUSTICE in editable mode

This step requires one terminal command. It makes the `justice`, `solvers`, and `analysis` packages importable.

1. In Anaconda Navigator → **Environments**, select `epa141a` → click **▶** → **Open Terminal**.
2. Navigate to the course repo root and run:

::::{tab-set}

:::{tab-item} macOS / Linux
```bash
cd /path/to/epa141a        # replace with your actual path
pip install -e JUSTICE-main --no-deps
```
:::

:::{tab-item} Windows
```bat
cd C:\path\to\epa141a      # replace with your actual path
pip install -e JUSTICE-main --no-deps
```
:::

::::

```{note}
The `--no-deps` flag prevents JUSTICE from overwriting the carefully pinned package versions
installed in Step 4 (especially `fair==2.1.3`).
```

Verify the install:
```bash
python -c "import justice; print('JUSTICE OK')"
# expected: JUSTICE OK
```

---

## Step 6 — Register the Jupyter kernel

Register the `epa141a` environment as a named kernel so notebooks can select it.

In the same terminal (with `epa141a` active):

::::{tab-set}

:::{tab-item} macOS / Linux
```bash
python -m ipykernel install --user \
    --name epa141a \
    --display-name "EPA141A (JUSTICE)"
```
:::

:::{tab-item} Windows
```bat
python -m ipykernel install --user ^
    --name epa141a ^
    --display-name "EPA141A (JUSTICE)"
```
:::

::::

You only need to do this once. The kernel persists across JupyterLab sessions.

---

## Step 7 — Launch JupyterLab

1. In Anaconda Navigator → **Home**.
2. Make sure the environment selector at the top shows **`epa141a`** (not `base`).
3. Click **Launch** under JupyterLab.
4. In the top-right kernel picker, select **`EPA141A (JUSTICE)`**.

```{admonition} Always launch from the repo root
:class: tip
JupyterLab must be launched from the `epa141a/` folder so that relative paths to
`JUSTICE-main/` resolve correctly. The setup cell in every notebook calls `os.chdir()` automatically.
```

---

## Step 8 — Verify the installation

Open `assignments_ema/assignment_01_exploratory_modeling.ipynb` and run **Exercise 0** (the first code cell). You should see:

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

All packages should show `✓`. A `RuntimeWarning: invalid value encountered in log` from the FaIR model is harmless — ignore it.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'justice'`

Either JUSTICE was not cloned or not installed. Check that a `JUSTICE-main/` folder exists in the repo root, then re-run Step 5.

### Kernel `EPA141A (JUSTICE)` not visible in JupyterLab

Re-run Step 6. Then restart JupyterLab.

### Kernel keeps dying immediately

JUSTICE loads FaIR climate data on startup and requires at least **2 GB free RAM**. Close other applications and try again.

### `ModuleNotFoundError: No module named 'ema_workbench'`

You are running the wrong kernel. Make sure you selected `EPA141A (JUSTICE)` in the top-right kernel picker — not the default base Python.

### `FileNotFoundError` for JUSTICE data files

Run the **setup cell** (first cell of every notebook) before any other cell. It calls `os.chdir()` to set the working directory to `JUSTICE-main/`.

### `reference_set_utilitarian.csv` not found (A7, A8)

You need to complete and save Assignment 6 first. The file is written to `model_answers_ema/results/`.

### Still stuck?

Post on the **Brightspace discussion board** with:
1. Your operating system and version.
2. Full error message (copy-paste, not a screenshot).
3. Output of `conda list` in the `epa141a` environment.

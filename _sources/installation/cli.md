# Command-Line Installation

This guide installs the course environment using terminal commands. It is **faster and more reliable** than the graphical method, especially on macOS and Linux.

```{admonition} Allow 10–20 minutes
:class: tip
A fresh install typically takes 10–20 minutes.
```

---

## Step 1 — Install Miniconda

If you already have `conda` available, skip to Step 2 below.

::::{tab-set}

:::{tab-item} macOS
```bash
# Apple Silicon (M1/M2/M3)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Intel Mac
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```
When asked *"Do you wish to run conda init?"* type **yes**. Open a **new** terminal window after installation.
:::

:::{tab-item} Windows
1. Download the [Miniconda Windows installer](https://docs.conda.io/en/latest/miniconda.html) (64-bit).
2. Run the `.exe` installer. Choose **"Just Me"** and accept the default path.
3. Open **Anaconda Prompt** from the Start Menu.
:::

:::{tab-item} Linux
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Open a **new** terminal after installation.
:::

::::

Verify:
```bash
conda --version
# expected: conda 24.x.x (or similar)
```

---

## Step 2 — Get the course repository

The course repository is at **[github.com/Hippo-Delft-AI-Lab/epa141a](https://github.com/Hippo-Delft-AI-Lab/epa141a)**.

::::{tab-set}

:::{tab-item} Git clone (recommended)
```bash
git clone https://github.com/Hippo-Delft-AI-Lab/epa141a.git
cd epa141a
```
:::

:::{tab-item} Download ZIP
1. Go to [github.com/Hippo-Delft-AI-Lab/epa141a](https://github.com/Hippo-Delft-AI-Lab/epa141a).
2. Click **Code → Download ZIP**, unzip, and navigate in:
```bash
cd path/to/epa141a
```
:::

::::

```{warning}
All subsequent commands must be run from the **repository root** — the folder containing
`assignments_ema/` and `requirements.txt`. Check with `ls` (macOS/Linux) or `dir` (Windows).
```

---

## Step 3 — Clone the JUSTICE model

JUSTICE is maintained in a **separate repository** at **[github.com/Hippo-Delft-AI-Lab/JUSTICE](https://github.com/Hippo-Delft-AI-Lab/JUSTICE)**. Clone it into the `epa141a/` root as `JUSTICE-main/`:

```bash
git clone https://github.com/Hippo-Delft-AI-Lab/JUSTICE.git JUSTICE-main
```

```{warning}
The folder **must** be named `JUSTICE-main/` — the assignment notebooks and the install command
in Step 5 both expect this exact name.
```

---

## Step 4 — Create the conda environment

From inside the `epa141a/` directory:

```bash
conda env create -f environment.yml
```

This reads `environment.yml` and installs all required packages. Allow **10–20 minutes**.

```{admonition} What gets installed
:class: dropdown

Python 3.12, JupyterLab ≥ 4.0, NumPy, Pandas, Matplotlib, SciPy, GeoPandas,
EMA Workbench ≥ 3.0, Platypus ≥ 1.1, FaIR 2.1.3, SALib ≥ 1.4, Plotly, ipyparallel, tqdm, kaleido.

`fair==2.1.3` is pinned exactly — do not upgrade it.
```

When complete you should see:
```
done
# To activate this environment, use:
#     $ conda activate epa141a
```

---

## Step 5 — Activate the environment

```bash
conda activate epa141a
```

Your prompt changes to `(epa141a)`. **You must activate this environment every time you open a new terminal before running any course code or launching JupyterLab.**

---

## Step 6 — Install JUSTICE in editable mode

Make the JUSTICE source importable. The `--no-deps` flag prevents overwriting the pinned versions installed in Step 4:

```bash
# Make sure (epa141a) is active and you are in the epa141a/ root
pip install -e JUSTICE-main --no-deps
```

```{note}
The `-e` flag installs in **editable mode** — any changes to the JUSTICE source are immediately
reflected without reinstalling. `--no-deps` protects the `fair==2.1.3` pin.
```

Verify:
```bash
python -c "import justice; print('JUSTICE OK')"
# expected: JUSTICE OK
```

---

## Step 7 — Register the Jupyter kernel

Register the environment as a named kernel so notebooks can select it:

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

You only need to do this once.

---

## Step 8 — Launch JupyterLab

```bash
jupyter lab
```

JupyterLab opens in your browser. Navigate to `assignments_ema/` or `labs/`, open a notebook, and select the **`EPA141A (JUSTICE)`** kernel in the top-right picker.

```{tip}
**VS Code users:** Open the repo folder in VS Code, open any `.ipynb` file, click the kernel
picker in the top right, and choose `EPA141A (JUSTICE)`. All assignments support VS Code natively —
the setup cell detects `__vsc_ipynb_file__` automatically.
```

---

## Step 9 — Verify the installation

Open `assignments_ema/assignment_01_exploratory_modeling.ipynb` and run **Exercise 0** (first code cell):

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

All packages should show `✓`. A `RuntimeWarning: invalid value encountered in log` from FaIR is harmless.

---

## Updating the environment

If `environment.yml` is updated during the course:

```bash
conda env update -f environment.yml --prune
```

---

## Useful conda commands

| Command | Purpose |
|---------|---------|
| `conda activate epa141a` | Activate the course environment |
| `conda deactivate` | Return to the base environment |
| `conda env list` | List all environments |
| `conda list` | List packages in the active environment |
| `conda env remove -n epa141a` | Delete and reinstall from scratch |

---

## Troubleshooting

### `conda env create` fails with a solve error

Try the faster `libmamba` solver:
```bash
conda install -n base conda-libmamba-solver
conda env create -f environment.yml --solver=libmamba
```

### `ModuleNotFoundError: No module named 'justice'`

Check that `JUSTICE-main/` exists in the repo root, then re-run Step 6.

### Kernel `EPA141A (JUSTICE)` not visible in JupyterLab

Re-run Step 7, then restart JupyterLab.

### Kernel keeps dying immediately

JUSTICE loads FaIR climate data on startup and requires at least **2 GB free RAM**. Close other applications.

### `FileNotFoundError` for JUSTICE data files

Always run the **setup cell** (first cell of every notebook) before any other cell. It calls `os.chdir()` to set the working directory to `JUSTICE-main/`.

### `RecursionError` in `perform_experiments` (Python 3.14 + tqdm)

A known incompatibility. The assignment notebooks include a fix (`tqdm.tqdm = tqdm.std.tqdm`) in the experiments cell. Make sure you have the latest notebooks and restart the kernel.

### `reference_set_utilitarian.csv` not found (A7, A8)

Complete and fully execute Assignment 6 first. The file is saved to `model_answers_ema/results/`.

### Slow first run in A8

A8 runs 28 policies × 50 FaIR scenarios (≈ 20–40 min, one time only). Results are cached to `results/reeval_utilitarian_28p_50s.npy` and loaded instantly on subsequent runs.

### Still stuck?

Post on the **Brightspace discussion board** with:
1. Your OS and version.
2. Full error message (copy-paste, not a screenshot).
3. Output of `conda list` in the `epa141a` environment.

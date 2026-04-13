# 🚀 Start Here

This page gets you from zero to running your first notebook in five steps. Follow them in order — each step takes only a few minutes.

```{admonition} One recommended path
:class: tip
This guide uses the terminal for reliability and VS Code as your editor. If you prefer a fully graphical approach, see the [Graphical Installation](installation/graphical.md) guide instead.
```

---

## Course repositories

Everything you need is in two public repositories on GitHub:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📦 epa141a
:link: https://github.com/Hippo-Delft-AI-Lab/epa141a
:link-type: url

Assignment notebooks, lab notebooks, `environment.yml`, configuration files.
+++
[→ github.com/Hippo-Delft-AI-Lab/epa141a](https://github.com/Hippo-Delft-AI-Lab/epa141a)
:::

:::{grid-item-card} 🌍 JUSTICE
:link: https://github.com/Hippo-Delft-AI-Lab/JUSTICE
:link-type: url

The climate Integrated Assessment Model used in all assignments and labs.
+++
[→ github.com/Hippo-Delft-AI-Lab/JUSTICE](https://github.com/Hippo-Delft-AI-Lab/JUSTICE)
:::

::::

Step 2 below shows you how to download both onto your laptop.

---

## How everything fits together

Before installing anything, here is the big picture:

```
GitHub (cloud)                  Your laptop
──────────────                  ────────────────────────────────
epa141a repo        clone  →    epa141a/
JUSTICE repo        clone  →    epa141a/JUSTICE-main/

                                conda creates:
environment.yml     ──────→     epa141a environment
                                (Python + all packages)

                                VS Code opens:
epa141a/            ──────→     .ipynb notebooks
                                runs in epa141a environment
```

---

## Step 1 — Install three tools

```{note}
**Before you start — system requirements:** ~2 GB free disk space (packages + FAIR climate data) and at least 2 GB of available RAM to run the JUSTICE model.
```

You need **Miniconda** (manages Python and packages), **VS Code** (your editor), and **Git** (downloads the course files).

**Install Miniconda:**

::::{tab-set}

:::{tab-item} macOS — Apple Silicon (M1/M2/M3/M4)
1. Download: [Miniconda macOS Apple Silicon](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.pkg)
2. Double-click the `.pkg` file and follow the installer.
3. Open a new terminal and verify:
```bash
conda --version
```
:::

:::{tab-item} macOS — Intel
1. Download: [Miniconda macOS Intel](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.pkg)
2. Double-click the `.pkg` file and follow the installer.
3. Open a new terminal and verify:
```bash
conda --version
```
:::

:::{tab-item} Windows
1. Download: [Miniconda Windows 64-bit](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)
2. Run the `.exe`. Choose **Just Me** and keep the default install path.
3. Open **Anaconda Prompt** from the Start Menu (use this instead of the regular terminal for all conda commands).
:::

:::{tab-item} Linux
1. Download the installer:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
2. Run it:
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
3. Accept the licence, keep the default install location, and say **yes** when asked to initialise conda.
4. Open a new terminal and verify:
```bash
conda --version
```
:::

::::

**Install VS Code:**

1. Go to [code.visualstudio.com](https://code.visualstudio.com) and download for your OS.
2. Run the installer. On Windows, tick **Add to PATH** when prompted.
3. Open VS Code, click the **Extensions** icon (left sidebar), and install:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)

**Install Git:**

::::{tab-set}

:::{tab-item} macOS
Run in a terminal:
```bash
xcode-select --install
```
A dialog will appear — click **Install**. Git is included in the Xcode Command Line Tools.
Verify: `git --version`
:::

:::{tab-item} Windows
1. Download the installer from [git-scm.com/download/win](https://git-scm.com/download/win).
2. Run it and accept the defaults.
3. Open a new **Anaconda Prompt** and verify: `git --version`
:::

:::{tab-item} Linux
```bash
# Ubuntu / Debian
sudo apt install git

# Fedora / RHEL
sudo dnf install git
```
Verify: `git --version`
:::

::::

---

## Step 2 — Get the course files

All course files live in two repositories on GitHub. Open a terminal (macOS/Linux) or Anaconda Prompt (Windows) and run:

```bash
# 1. Clone the course repository
git clone https://github.com/Hippo-Delft-AI-Lab/epa141a.git

# 2. Move into it
cd epa141a

# 3. Clone JUSTICE inside it — the folder name must be JUSTICE-main
git clone https://github.com/Hippo-Delft-AI-Lab/JUSTICE.git JUSTICE-main
```

After this your folder structure should look like:

```
epa141a/
├── JUSTICE-main/       ← climate model
├── assignments_ema/    ← your assignment notebooks
├── labs/               ← lab notebooks
└── environment.yml     ← package list
```

```{warning}
The `JUSTICE-main/` folder must have that exact name. The notebooks and install command in Step 3 both depend on it.
```

---

## Step 3 — Create the Python environment

Still in the `epa141a/` folder, run:

```bash
conda env create -f environment.yml
```

This reads `environment.yml` and installs Python 3.12 plus all course packages. **Allow 15–30 minutes** — the packages are large.

When it finishes you should see:
```
done
# To activate this environment, use:
#     $ conda activate epa141a
```

```{admonition} If the install fails with a solve error
:class: dropdown
Try the faster libmamba solver:
```bash
conda install -n base conda-libmamba-solver
conda env create -f environment.yml --solver=libmamba
\`\`\`
```

---

## Step 4 — Install JUSTICE and register the kernel

Activate the environment, then run two commands:

```bash
# Activate the environment
conda activate epa141a

# Install JUSTICE so notebooks can import it
pip install -e JUSTICE-main --no-deps

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name epa141a --display-name "EPA141A (JUSTICE)"
```

Verify JUSTICE is working:
```bash
python -c "import justice; print('JUSTICE OK')"
# expected output: JUSTICE OK
```

You only need to do Steps 3 and 4 once. From here on, you just open VS Code.

---

## Step 5 — Open your first notebook

1. Open VS Code.
2. Go to **File → Open Folder…** and select the `epa141a/` folder.
3. In the Explorer panel, open `assignments_ema/assignment_01_exploratory_modeling.ipynb`.
4. In the **top-right corner** of the notebook, click the kernel picker and select **`epa141a`** (or `EPA141A (JUSTICE)`).
5. Run the first cell (**Exercise 0**) by pressing `Shift Enter`.

You should see:

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

Every item should show `✓`. If anything shows `✗`, see the [Troubleshooting](#troubleshooting) section below.

---

## You are set up. What next?

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 📅 Check the schedule
See what's due this week.
+++
[→ Schedule](schedule.md)
:::

:::{grid-item-card} 🔬 Start Lab 1
Warm up with the Lake Problem before Assignment 1.
+++
[→ Lab 1](labs/lab_01.md)
:::

:::{grid-item-card} 📓 Open Assignment 1
Begin the first exploratory modelling assignment.
+++
[→ Assignment 1](assignments/assignment_01.md)
:::

::::

---

## Troubleshooting

### `conda` is not recognised

Close the terminal, open a new one, and try again. On macOS, make sure you opened a **new** terminal window after installing Miniconda. On Windows, use **Anaconda Prompt**, not the regular terminal.

### `git` is not recognised

Git may not be installed. See the Git installation instructions in Step 1 above.

### `ModuleNotFoundError: No module named 'justice'`

Check that `JUSTICE-main/` exists inside `epa141a/`, then re-run:
```bash
conda activate epa141a
pip install -e JUSTICE-main --no-deps
```

### Kernel `EPA141A (JUSTICE)` not visible in VS Code

Re-run the `ipykernel install` command from Step 4, then reload VS Code (`Ctrl/⌘ Shift P` → **Developer: Reload Window**).

### `✗` next to a package in Exercise 0

You are likely running the wrong kernel. Click the kernel picker in the top-right and make sure it shows `epa141a` or `EPA141A (JUSTICE)` — not `base` or any other environment.

### Kernel keeps dying or restarting

JUSTICE loads the full FaIR climate dataset on startup and needs at least **2 GB of free RAM**. Close other applications and try again. If it still crashes, restart your computer to free memory.

### `FileNotFoundError` when running JUSTICE cells

Every assignment notebook starts with a **setup cell** (clearly labelled) that handles paths. You must run this cell before any other cell — it calls `os.chdir()` to point JUSTICE at its data files. If you skip it or run cells out of order, all subsequent JUSTICE calls will fail with a `FileNotFoundError`.

### `RuntimeWarning: invalid value encountered in log` from FAIR

This warning is **harmless**. FaIR catches it internally and the results are still valid. You can safely ignore it.

### `RecursionError` in `perform_experiments`

This is a known incompatibility between Python 3.14 and `tqdm ≥ 4.67` inside Jupyter. Add this line to the top of the cell that calls `perform_experiments`, then restart the kernel and re-run:

```python
import tqdm; tqdm.tqdm = tqdm.std.tqdm
```

### Still stuck?

Post on the **Brightspace discussion board** with your OS, the full error message (copy-paste), and the output of `conda list` in the `epa141a` environment.

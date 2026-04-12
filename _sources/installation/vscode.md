# VS Code Setup

This guide sets up **Visual Studio Code** as your editor for the course. VS Code supports Jupyter notebooks natively and integrates well with conda environments — no browser required.

```{admonition} Already installed conda?
:class: tip
This guide assumes you have already created the `epa141a` conda environment. If not, complete the [Graphical](graphical.md) or [Command-Line](cli.md) installation first, then return here.
```

---

## Step 1 — Install VS Code

Go to [code.visualstudio.com](https://code.visualstudio.com) and download the installer for your operating system.

::::{tab-set}

:::{tab-item} macOS
1. Download the `.zip` file and extract it.
2. Drag **Visual Studio Code.app** to your **Applications** folder.
3. Open VS Code from Launchpad or Spotlight (`⌘ Space` → type `code`).

```{tip}
To open VS Code from the terminal, open the Command Palette (`⌘ Shift P`) → type `Shell Command: Install 'code' command in PATH` → press Enter. You can then type `code .` in any folder to open it in VS Code.
```
:::

:::{tab-item} Windows
1. Run the downloaded `.exe` installer.
2. On the **Select Additional Tasks** screen, tick:
   - **Add "Open with Code" action to Windows Explorer file context menu**
   - **Add to PATH**
3. Finish the installation and launch VS Code from the Start Menu.
:::

:::{tab-item} Linux
```bash
# Debian / Ubuntu
sudo apt install ./<downloaded-file>.deb

# Or via snap
sudo snap install --classic code
```
:::

::::

---

## Step 2 — Install the Python and Jupyter extensions

VS Code needs two extensions to work with conda environments and Jupyter notebooks.

1. Click the **Extensions** icon in the left sidebar (or press `Ctrl/⌘ Shift X`).
2. Search for and install each of the following:

| Extension | Publisher | What it does |
|-----------|-----------|-------------|
| **Python** | Microsoft | Conda/virtual environment support, linting, IntelliSense |
| **Jupyter** | Microsoft | Run `.ipynb` notebooks natively in VS Code |

```{note}
Installing the **Jupyter** extension also installs **Jupyter Keymap** and **Jupyter Notebook Renderers** automatically.
```

---

## Step 3 — Open the course repository

::::{tab-set}

:::{tab-item} From the terminal
Navigate to the `epa141a/` folder and open VS Code:
```bash
cd path/to/epa141a
code .
```
:::

:::{tab-item} From VS Code
1. Go to **File → Open Folder…**
2. Navigate to and select the `epa141a/` folder (the one containing `assignments_ema/` and `environment.yml`).
3. Click **Open**.
:::

::::

```{warning}
Always open the **`epa141a/` root folder**, not a subfolder. Relative paths to `JUSTICE-main/` in the notebooks will break if VS Code is opened from the wrong directory.
```

---

## Step 4 — Select the conda environment as your Python interpreter

Tell VS Code to use the `epa141a` conda environment:

1. Open the Command Palette (`Ctrl/⌘ Shift P`).
2. Type **Python: Select Interpreter** and press Enter.
3. From the list, select the entry that shows **`epa141a`** — it will look like:

```
Python 3.12.x ('epa141a': conda)
~/miniconda3/envs/epa141a/bin/python
```

```{tip}
If `epa141a` does not appear in the list, click **Enter interpreter path…** and paste the full path:

- **macOS/Linux**: `~/miniconda3/envs/epa141a/bin/python`
- **Windows**: `C:\Users\<you>\miniconda3\envs\epa141a\python.exe`
```

---

## Step 5 — Open a notebook and select the kernel

1. In the **Explorer** panel (left sidebar), navigate to `assignments_ema/` and open `assignment_01_exploratory_modeling.ipynb`.
2. VS Code will open the notebook in its built-in editor.
3. In the **top-right corner**, click the kernel picker (it may show **Select Kernel** or a Python version).
4. Choose **Python Environments…** → select **`epa141a`**.

The kernel picker should now show **`EPA141A (JUSTICE)`** or **`epa141a (Python 3.12.x)`**.

```{warning}
Always confirm the kernel in the top-right before running any cells. Running with the wrong kernel (e.g. base Python) will cause `ModuleNotFoundError` for `justice`, `ema_workbench`, and other course packages.
```

---

## Step 6 — Run the verification cell

1. Open `assignments_ema/assignment_01_exploratory_modeling.ipynb`.
2. Run **Exercise 0** (the first code cell) by clicking **▶** or pressing `Shift Enter`.
3. You should see:

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

## Your daily workflow

Once the environment is installed, this is what you do every time you sit down to work:

1. **Open VS Code** and open the `epa141a/` folder (**File → Open Folder…**).
2. **Navigate** to `assignments_ema/` or `labs/` in the Explorer panel and open a notebook.
3. **Confirm the kernel** in the top-right corner shows `epa141a` — change it if not.
4. **Run the setup cell first** — every notebook starts with a setup cell that sets the working directory and imports packages. Always run it before any other cell.
5. **Run cells in order**, top to bottom. Skipping cells or running them out of order is the most common source of errors.
6. **Save regularly** (`Ctrl/⌘ S`). VS Code auto-saves, but saving also checkpoints your notebook output.

```{warning}
If you close and reopen a notebook, the kernel loses all its state — variables, imports, and results are gone. Always re-run from the top (or use **Run All**) before continuing.
```

---

## Working with Jupyter notebooks

A Jupyter notebook (`.ipynb`) is made of **cells**. There are two types:

| Cell type | What it contains | How to run |
|-----------|-----------------|------------|
| **Code** | Python code | `Shift Enter` or click ▶ |
| **Markdown** | Explanatory text, instructions | `Shift Enter` to render |

**Understanding the cell status indicator:**

- `[ ]` — not yet run
- `[*]` — currently running (be patient)
- `[4]` — ran successfully (number = order it was run)
- Red border or traceback below — an error occurred

**The kernel** is the Python engine running behind your notebook. Think of it as the "brain" that remembers everything you have run so far. If the kernel crashes or you restart it, all variables are cleared.

**When to restart the kernel:**

- A cell is stuck on `[*]` for much longer than expected and does not respond to interruption
- You are getting unexpected errors and want a clean slate
- You want to verify your notebook runs correctly from top to bottom

To restart: click the **⟳ Restart** button in the notebook toolbar, or press `Ctrl/⌘ Shift P` → **Notebook: Restart Kernel**. Then re-run all cells from the top.

---

## Getting updates from the course repository

If the course team pushes corrections or new files to the repository after you have cloned it, you can pull the updates without re-downloading everything.

Open a terminal in the `epa141a/` root folder (`` Ctrl/⌘ ` `` in VS Code) and run:

```bash
git pull
```

```{admonition} If you downloaded a ZIP instead of cloning
:class: warning
ZIP downloads do not track the remote repository — you cannot `git pull`. You will need to download a new ZIP and manually copy any changed files. We recommend cloning with `git clone` to avoid this.
```

If `git pull` reports a conflict (you edited a file that was also updated on the server), post on the Brightspace discussion board before doing anything — do not force-overwrite your work.

---

## Long-running cells

Some assignments involve computationally intensive simulations. Knowing what to expect will save you a lot of anxiety.

| Assignment | What runs | Typical duration |
|------------|-----------|-----------------|
| A1 – A3 | Sampling, sensitivity analysis | seconds to a few minutes |
| A4 | Problem formulation only | seconds |
| A5 | MOEA optimisation (local, 5 000 NFE) | 5–15 minutes |
| A5 | MOEA optimisation (local, 100 000 NFE) | 30–60 minutes |
| A6 | Multi-seed MOEA convergence | 20–40 minutes per seed |
| A7 | Pareto visualisation | seconds to minutes |
| A8 | Re-evaluation across 50 scenarios | 20–40 minutes (runs once, then cached) |

**What to do while a cell is running:**

- Leave `[*]` alone — it is working.
- Do not close VS Code or put your laptop to sleep.
- You can read, work on markdown cells, or review earlier results — just do not run other code cells in the same notebook.

**How to safely interrupt a long-running cell:**

Click the **⬛ Stop** button in the notebook toolbar, or press `I` twice in command mode. The cell will stop and show an error — this is expected. You can then restart and re-run from that point.

```{tip}
The most computationally expensive runs (A5 100k NFE, A6 multi-seed) are designed to be started and left running. Plan to run them when you do not need your computer for an hour.
```

---

## Useful VS Code shortcuts for notebooks

| Shortcut | Action |
|----------|--------|
| `Shift Enter` | Run current cell and move to next |
| `Ctrl/⌘ Enter` | Run current cell and stay |
| `A` | Insert cell above (in command mode) |
| `B` | Insert cell below (in command mode) |
| `DD` | Delete current cell (in command mode) |
| `Esc` | Enter command mode |
| `Enter` | Enter edit mode |
| `Ctrl/⌘ Shift P` | Open Command Palette |

```{tip}
Press `Esc` to enter **command mode** (blue cell border) before using single-key shortcuts like `A`, `B`, or `DD`.
```

---

## Troubleshooting

### `epa141a` environment not listed in the interpreter picker

Make sure the conda environment was created successfully:
```bash
conda env list
```
You should see `epa141a` in the list. If not, re-run the [Graphical](graphical.md) or [CLI](cli.md) installation guide.

### Kernel picker shows no conda environments

Reload VS Code (`Ctrl/⌘ Shift P` → **Developer: Reload Window**). VS Code sometimes needs a restart after a new conda environment is created.

### `ModuleNotFoundError: No module named 'justice'`

Check that `JUSTICE-main/` exists inside the `epa141a/` folder, then re-run the editable install step from the terminal inside VS Code (`Ctrl/⌘ \`\`):
```bash
conda activate epa141a
pip install -e JUSTICE-main --no-deps
```

### `FileNotFoundError` for JUSTICE data files

Always run the **setup cell** (first cell of every notebook) before any other cell. It calls `os.chdir()` to set the working directory to `JUSTICE-main/`.

### Still stuck?

Post on the **Brightspace discussion board** with:
1. Your OS and version.
2. Full error message (copy-paste, not a screenshot).
3. Output of `conda list` in the `epa141a` environment.

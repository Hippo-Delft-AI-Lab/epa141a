# Assignment 5 — Many-Objective Optimisation (Local Run)

**Key methods:** Borg MOEA · ε-NSGA-II · Platypus · Pareto-approximate set

---

## Learning objectives

1. Configure a **Many-Objective Evolutionary Algorithm** (ε-NSGA-II via Platypus/Borg) for the JUSTICE problem defined in Assignment 4.
2. Run optimisation **locally** using `ema_workbench`.
3. Inspect and plot the resulting **Pareto-approximate set** in objective space.
4. Understand the role of: population size, number of function evaluations (NFE), and ε values.
5. Save Pareto results to `results/` for convergence analysis in Assignment 6.

## Prerequisites

- **Assignment 4 completed** — the problem formulation (`config_student.json`) is required.

```{warning}
This assignment involves running an MOEA on JUSTICE. Allow **1–2 hours** for the optimisation
to complete on a standard laptop. Start early and do not close your computer during the run.
Results are saved automatically so you do not lose progress if you need to pause.
```

## Notebook

`assignments_ema/assignment_05_moea_local.ipynb`

---

## Running on DelftBlue HPC (recommended for 100k+ NFE)

Running 5 seeds × 50 000 NFE locally takes roughly **one working day** and leaves your laptop unusable in the meantime. DelftBlue runs all 5 seeds **in parallel** and finishes in about **2–4 hours**.

```{admonition} You do not need HPC to pass this assignment
:class: tip
The local run (1 seed, 50 000 NFE) is sufficient for submission. HPC is recommended if you want a complete 5-seed reference set for Assignments 6 and 7, or if your laptop is slow.
```

### Step 1 — Get access to DelftBlue

1. Go to [accounts.login.delftblue.tudelft.nl](https://accounts.login.delftblue.tudelft.nl) and log in with your TU Delft NetID.
2. Request access to the **education-tpm-msc-epa** account — your TA will confirm when it is active.
3. Follow the [DelftBlue SSH setup guide](https://doc.dhpc.tudelft.nl/delftblue/Connecting-to-DelftBlue/) to set up your connection.

### Step 2 — Transfer your files to DelftBlue

From your laptop, copy the entire course folder to your DelftBlue home directory:

::::{tab-set}

:::{tab-item} macOS / Linux
```bash
# From your laptop terminal
scp -r /path/to/epa141a <netid>@login.delftblue.tudelft.nl:~/epa141a
```
:::

:::{tab-item} Windows
Use [WinSCP](https://winscp.net) or [MobaXterm](https://mobaxterm.mobatek.net) to transfer the `epa141a/` folder to your home directory on DelftBlue.
:::

::::

```{warning}
Make sure `config/config_student.json` (created in Assignment 4) is included in the transfer — the optimisation script reads it at runtime.
```

### Step 3 — Set up the conda environment on DelftBlue

SSH into DelftBlue and create the environment:

```bash
ssh <netid>@login.delftblue.tudelft.nl

# Navigate to your course folder
cd ~/epa141a

# Load miniconda and create the environment
module load 2023r1
module load miniconda3
conda env create -f environment.yml

# Install JUSTICE
conda activate epa141a
pip install -e JUSTICE-main --no-deps
```

You only need to do this once.

### Step 4 — Submit the job array

From the `epa141a/` root on DelftBlue:

```bash
sbatch assignments_ema/submit_to_delftblue.sh
```

This submits 5 jobs — one per seed — that run in parallel. Each job uses 20 CPU cores and runs for up to 4 hours.

**Monitor your jobs:**
```bash
squeue -u $USER          # see job status
cat results/logs/seed_1_<jobid>.out  # see output for seed 1
```

**Expected output** once jobs complete:
```
results/
└── UTILITARIAN_50000_9845531/
│   ├── pareto_front_9845531.csv
│   ├── UTILITARIAN_50000_9845531.tar.gz
│   └── convergence_9845531.csv
└── UTILITARIAN_50000_1644652/
    ├── ...
```

### Step 5 — Download results to your laptop

Once all jobs show `COMPLETED` in `squeue`, download the results:

::::{tab-set}

:::{tab-item} macOS / Linux
```bash
# From your laptop terminal
scp -r <netid>@login.delftblue.tudelft.nl:~/epa141a/results ./results
```
:::

:::{tab-item} Windows
Use WinSCP or MobaXterm to download the `results/` folder from `~/epa141a/results` on DelftBlue into your local `epa141a/results/`.
:::

::::

Open your Assignment 5 notebook locally, run Step 3 (Load and Inspect Results), and verify all 5 seeds loaded correctly.

### Troubleshooting HPC runs

**`conda: command not found` in the job script**
Make sure `module load miniconda3` is in your SLURM script and that the environment name matches (`epa141a`).

**`sbatch: error: Invalid account`**
Your access to `education-tpm-msc-epa` is not yet approved. Contact your TA.

**Job stays in `PENDING` state**
The cluster is busy — this is normal. Check estimated start time with `squeue --start -u $USER`.

**`FileNotFoundError: config_student.json`**
The config file was not transferred. Copy it from your laptop:
```bash
scp config/config_student.json <netid>@login.delftblue.tudelft.nl:~/epa141a/config/
```

---

## Submission

Submit `A5_<studentnumber>.ipynb` via Brightspace with all cells executed **and Pareto results saved** to `results/`.

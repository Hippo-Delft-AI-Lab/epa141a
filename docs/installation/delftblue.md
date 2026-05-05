# Running on DelftBlue (HPC)

DelftBlue is TU Delft's high-performance computing cluster. You will need it for **Assignment 5** when your local machine cannot run enough function evaluations in reasonable time.

```{admonition} When to use DelftBlue
:class: tip
Run Assignment 5 locally first with a small NFE (e.g. 1 000–5 000) to verify everything works. Switch to DelftBlue when you need NFE ≥ 10 000 or want to run 5 seeds in parallel without tying up your laptop for hours.
```

---

## 1. Connect to DelftBlue

Open a terminal and SSH into the login node using your NetID:

```bash
ssh netid@login.delftblue.tudelft.nl
```

Replace `netid` with your TU Delft NetID. You will be prompted for your TU Delft password.

```{note}
You must be on the TU Delft network or connected via **EduVPN** before SSH will work.
Download EduVPN from [tudelft.nl/en/it/services/eduvpn](https://www.tudelft.nl/en/it/services/eduvpn).
```

---

## 2. Transfer your files

From your **local machine** (not inside the SSH session), copy your `epa141a/` folder to DelftBlue:

```bash
rsync -avz --exclude '.git' --exclude '_build' \
    /path/to/epa141a/ \
    netid@login.delftblue.tudelft.nl:~/epa141a/
```

Replace `/path/to/epa141a/` with the actual path on your machine. Run this command again after making any local changes to sync updates.

```{tip}
On Windows, use **MobaXterm** or **WinSCP** for file transfer if you are not comfortable with the terminal.
```

---

## 3. Set up the environment

The `epa141a` conda environment is already configured on DelftBlue for this course. Activate it:

```bash
module load 2023r1
module load miniconda3
conda activate epa141a
```

Verify the environment is working:

```bash
python -c "import justice; import ema_workbench; print('OK')"
```

---

## 4. Submit Assignment 5

A ready-to-use SLURM submission script is included in the repo at `assignments_ema/submit_to_delftblue.sh`. It runs **5 independent seeds in parallel**, each on 20 CPUs for up to 4 hours.

From the `epa141a/` root on DelftBlue, submit the job array:

```bash
cd ~/epa141a
sbatch assignments_ema/submit_to_delftblue.sh
```

The script uses the following settings — you do not need to change them unless your TA says otherwise:

| Setting | Value |
|---------|-------|
| Seeds | 5 parallel (pre-defined) |
| NFE per seed | 50 000 |
| Ensemble members | 50 |
| CPUs per seed | 20 |
| Time limit | 4 hours |
| SLURM account | `education-tpm-msc-epa` |

Results are written to `epa141a/results/` and logs to `epa141a/results/logs/`.

---

## 5. Monitor your jobs

Check the status of your submitted jobs:

```bash
squeue -u $USER
```

| Status | Meaning |
|--------|---------|
| `PD` | Pending — waiting in the queue |
| `R` | Running |
| `CG` | Completing |

To cancel all your running jobs:

```bash
scancel -u $USER
```

---

## 6. Retrieve your results

Once the jobs finish, copy the results back to your local machine. Run this from your **local terminal**:

```bash
rsync -avz \
    netid@login.delftblue.tudelft.nl:~/epa141a/results/ \
    /path/to/epa141a/assignments_ema/results/
```

Then open Assignment 6 locally — it will load the results from `assignments_ema/results/` as usual.

---

## Troubleshooting

**Job fails immediately**
Check the error log: `cat results/logs/seed_1_<jobid>.err`. The most common cause is a missing `config/config_student.json` — make sure you completed Assignment 4 locally and transferred the file.

**`conda activate` not found**
Run `module load miniconda3` before activating the environment.

**Permission denied on SSH**
Make sure you are connected to EduVPN and using your correct NetID and TU Delft password.

**Quota exceeded**
DelftBlue home directories have limited space. Delete old result archives (`*.tar.gz`) once you have retrieved the Pareto-front CSVs you need.

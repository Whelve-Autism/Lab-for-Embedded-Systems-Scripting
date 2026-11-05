<div align="center">

# Lab for Embedded Systems Scripting

**Differential Evolution · Comparative Experiments · Linux Automation**

**English** · [简体中文](./README.zh-CN.md)

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![Benchmark](https://img.shields.io/badge/Benchmark-CEC2014-005A9C)
![Automation](https://img.shields.io/badge/Automation-Bash%20%2B%20Cron-4EAA25?logo=gnubash&logoColor=white)

From implementing a basic optimizer to comparing improved algorithms and automating experiments on Linux.

</div>

---

**Explore:** [Overview](#overview) · [Setup](#setup) · [Project 1](#project-1) · [Project 2](#project-2) · [Project 3](#project-3) · [Outputs](#outputs)

<a id="overview"></a>
## Overview

This repository contains three connected course projects for **Embedded Systems Scripting**. Python implements and evaluates Differential Evolution (DE) algorithms; Bash and Cron automate the experimental workflow.

| Project | Course objective | What this repository provides |
| :--- | :--- | :--- |
| [Project&nbsp;1](./project_1/) | Implement basic DE and evaluate repeated runs | DE/rand/1/bin, CEC2014 experiments, statistical summaries, and convergence plots |
| [Project&nbsp;2](./project_2/) | Implement improved DE methods, develop a combined method, and evaluate its strategies | Six-method comparison, NICE-DE, four ablation variants, and comparison plots |
| [Project&nbsp;3](./project_3/) | Automate the Project 2 application and analyze its logs on Linux | Parameter sweeps, isolated experiment batches, grep/awk summaries, and hourly Cron scheduling |

> **Project 1** Basic DE → **Project 2** Comparison & ablation → **Project 3** Automation

The repository publishes **source code, configuration, and usage documentation**. Generated experiment results, figures, execution logs, and course evidence remain local.

### Repository layout

```text
.
├── README.md                       # English
├── README.zh-CN.md                 # 简体中文
├── project_1/
│   ├── de.py                       # Basic DE/rand/1/bin
│   ├── run_cec2014.py               # Repeated benchmark experiments
│   ├── plot_convergence.py
│   └── schwefel_patch.py
├── project_2/
│   ├── common/                     # Shared experiment harness and benchmark patch
│   ├── methods/                    # Baselines, NICE-DE, and ablations
│   └── plot_convergence.py
└── project_3/
    ├── run_experiments.sh           # Bash orchestration and result extraction
    ├── run_nice_p3.py               # Entry point reusing Project 2 NICE-DE
    ├── install_env.sh
    ├── run_nohup.sh
    ├── check_cron_evidence.sh
    ├── crontab.example
    └── USAGE.txt
```

<a id="setup"></a>
## Setup

Clone the repository and create a Python environment. The following commands are for macOS/Linux shells and are run from the repository root:

```bash
git clone https://github.com/Whelve-Autism/Lab-for-Embedded-Systems-Scripting.git
cd Lab-for-Embedded-Systems-Scripting

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r project_2/requirements.txt scipy cma matplotlib
```

| Package | Purpose |
| :--- | :--- |
| NumPy, pandas | Numerical computation and result tables |
| opfunu 1.0.1 | CEC2014 benchmark functions |
| SciPy | DE-VCR and ECL-SPACMA components |
| cma | NICE-DE local CMA-ES search |
| Matplotlib | Convergence plots and PDF export |

The project requirements retain `setuptools<81` for compatibility with the benchmark dependency. The additional packages in the command cover the method-specific and plotting imports. Project 3 also provides its own environment installer, described below.

### Default evaluation protocol

| Setting | Projects 1 & 2 | Project 3 |
| :--- | :--- | :--- |
| Benchmark | CEC2014, F1–F30 | CEC2014, F1–F30 |
| Dimension / bounds | 10 / [−100, 100] | 10 / [−100, 100] |
| Evaluation budget | 100,000 per function and seed | 20,000 / 50,000 / 100,000 per function |
| Random seeds | 1001–1020: 20 runs per function | 1001: one run per function and configuration |
| Reported error | `f(x_best) - f*` | Mean of `abs(f(x_best) - f*)` across functions |
| Population | Project 1: 100; Project 2: method-specific | NICE-DE: 100 |

Budgets count **objective-function evaluations (FEs)**. Population settings and adaptation policies are defined in each method; they should be recorded alongside comparison results. Comparisons with an original paper also need that paper's benchmark, dimension, and experimental settings.

<a id="project-1"></a>
## Project 1 · Basic Differential Evolution

[Source directory](./project_1/) · [Experiment runner](./project_1/run_cec2014.py)

Implements **DE/rand/1/bin** with population size 100, mutation factor 0.5, and crossover rate 0.9. Each CEC2014 function is evaluated over 20 seeded runs.

```bash
python project_1/run_cec2014.py --workers 8
python project_1/plot_convergence.py
```

Results include **Best, Median, Mean, Worst, and Std**, together with per-run fitness errors and convergence checkpoints. `Std` is the sample standard deviation; its square gives the sample variance when needed for the report.

The plotting script exports 30 PDFs under `project_1/out/plot/`, using the median error curve and an interquartile band. Both projects' plotting scripts request **Times New Roman**; make that font available on the plotting machine to preserve the intended appearance.

<a id="project-2"></a>
## Project 2 · Improved DE and Strategy Evaluation

[Source directory](./project_2/) · [Shared experiment harness](./project_2/common/experiment.py)

### Main comparison

| Method | Role | Entry point |
| :--- | :--- | :--- |
| BIDE | Basic DE/rand/1/bin baseline | [run.py](./project_2/methods/BIDE/run.py) |
| DLS-DE | Improved DE baseline | [run.py](./project_2/methods/DLS_DE/run.py) |
| DE-VCR | Improved DE baseline | [run.py](./project_2/methods/DE_VCR/run.py) |
| ECL-SPACMA | Hybrid baseline with SLSQP local search in this implementation | [run.py](./project_2/methods/ECL_SPACMA/run.py) |
| iStratDE | NumPy implementation of individual-level strategy diversity | [run.py](./project_2/methods/iStratDE/run.py) |
| **NICE-DE** | Team-developed method combining adaptive DE and niche-guided CMA-ES search | [run.py](./project_2/methods/NICE_DE/run.py) |

Run the six-method comparison from the repository root:

```bash
for method in BIDE DLS_DE DE_VCR ECL_SPACMA iStratDE NICE_DE; do
  python "project_2/methods/${method}/run.py" --workers 8
done
```

### NICE-DE and ablation experiments

**NICE-DE** stands for *Niche-Informed Covariance-Enhanced Differential Evolution*. It combines adaptive DE global exploration, niche-guided persistent CMA-ES local search, and information exchange between the two. Local-search evaluations count toward the same total FE budget.

| Variant directory | Mechanism removed |
| :--- | :--- |
| [`NICE_wo_Niche`](./project_2/methods/NICE_wo_Niche/) | Niche-guided selection of local-search regions |
| [`NICE_wo_Persistence`](./project_2/methods/NICE_wo_Persistence/) | CMA-ES state persistence across generations |
| [`NICE_wo_Collaboration`](./project_2/methods/NICE_wo_Collaboration/) | CMA-to-DE injection and CMA contribution to the reported global best |
| [`NICE_wo_LocalSearch`](./project_2/methods/NICE_wo_LocalSearch/) | CMA-ES local search; the available FE budget is used by DE |

```bash
for variant in NICE_wo_Niche NICE_wo_Persistence NICE_wo_Collaboration NICE_wo_LocalSearch; do
  python "project_2/methods/${variant}/run.py" --workers 8
done

python project_2/plot_convergence.py --pack all
```

The plotting command reads the main-comparison and ablation CSVs, then creates matching PDF figure sets under `project_2/out/plot/main/` and `project_2/out/plot/ablation/`. Per-run CSVs also supply the data for the report's statistical tables and ranks.

<a id="project-3"></a>
## Project 3 · Bash Automation and Experiment Analysis

[Source directory](./project_3/) · [Detailed usage](./project_3/USAGE.txt) · [Cron configuration](./project_3/crontab.example)

Project 3 reuses the **Project 2 NICE-DE implementation** through a thin Python entry point. A Bash loop varies the FE budget, appends program output to a log, and extracts one numerical score per configuration with `grep | awk`.

### Run on Linux

Keep `project_2` and `project_3` as sibling directories. The installer creates an environment under `project_3/.venv`:

```bash
cd project_3
chmod +x install_env.sh run_experiments.sh run_nohup.sh check_cron_evidence.sh
./install_env.sh
./run_experiments.sh
```

Defaults: **three FE budgets, 30 functions, seed 1001, and 16 worker processes**. NICE-DE runs on the CPU. A `CUDA_VISIBLE_DEVICES` setting does not enable GPU computation.

For a smaller pipeline check, use a distinct manual batch:

```bash
FE_LIST="2000" FUNCS="1-2" WORKERS=2 ./run_experiments.sh
```

This subset check does not replace the full course experiments or the required scheduled executions.

### Scalar score and log extraction

For each FE budget, the runner evaluates every required function once and computes:

$$
E_{\mathrm{mean}} = \frac{1}{30}\sum_{i=1}^{30}\left|f_i(x_{\mathrm{best},i}) - f_i^{\star}\right|.
$$

For a function subset, the denominator is the number of requested functions. A valid score requires every requested function to succeed.

| Record or file | Purpose |
| :--- | :--- |
| `BATCH_START` / `BATCH_END` | Timestamped batch boundaries and completion status |
| `RUN_START` / `RUN_END` | Configuration identifiers, parameters, and exit status |
| `P3_RESULT` | Machine-readable score, function count, and failure count |
| `experiment.log` | Append-only Python output and experiment records |
| `summary_report.txt` | Latest batch's extracted scores and corresponding raw result lines |

Each invocation creates a unique batch directory and preserves its own summary. The latest summary filters by the current batch ID, so earlier log records are not mixed into the new table. Invalid inputs and failed configurations produce a nonzero exit status; failed configurations are not reported as valid partial averages.

### Hourly Cron scheduling

Edit the absolute paths in [crontab.example](./project_3/crontab.example), then install the schedule with `crontab -e`. A portable example is:

```cron
0 * * * * PYTHON_BIN="/absolute/path/project_3/.venv/bin/python" WORKERS=16 /bin/bash "/absolute/path/project_3/run_experiments.sh" >> "/absolute/path/project_3/cron_execution.log" 2>&1
```

Keep Linux running until **at least three scheduled batches have completed successfully**. Verify distinct batch IDs, timestamps, saved results, and successful `BATCH_END` records. Manual runs and subset checks do not count toward the scheduled-run requirement. Repeating seed 1001 verifies scheduled execution; it does not add new independent random seeds.

For the course submission, retain the Cron configuration, script, timestamped logs, result summaries, and actual terminal screenshots. Include command-line editor use, executable permissions, help/manual resources, input/error checks, and an explanation of the log format. See [USAGE.txt](./project_3/USAGE.txt) for these commands and manual restart instructions. Disable the test schedule after collecting the required evidence.

<a id="outputs"></a>
## Outputs and Repository Scope

The files below are generated locally and are excluded from Git:

| Location | Contents |
| :--- | :--- |
| `project_1/out/` | Per-run CSV, statistical summary, and convergence PDFs |
| `project_2/out/<method>/` | Per-method results, protocol, curves, and available diagnostics |
| `project_2/out/plot/` | Main-comparison and ablation PDFs |
| `project_3/results/P3/<batch_id>/` | Per-budget results and a preserved batch summary |
| Project 3 logs, `archive/`, and `screenshots/` | Execution records and course evidence |
| `.venv/`, `__pycache__/` | Local environments and Python caches |

Project 1/2 runners write to fixed output locations. Project 3 preserves separate batch directories; its `--clean` option archives the aggregate logs **and starts a new batch**. Do not use `--clean` in the hourly Cron job.

The empty `.gitkeep` files only retain the Project 3 results/evidence directory structure. Course reports and original reference PDFs are maintained separately from this source repository.

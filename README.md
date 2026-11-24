# Revisiting ADFL: An Adversarial Evaluation of GAN-Based Distillation Defenses for Federated Backdoors

This repository contains the code, checkpoints, and results for a re-implementation and
adversarial evaluation of **ADFL** (Zhu et al., 2023) — a server-side federated-learning
backdoor defense that seeds a GAN discriminator from the backdoored model, labels generated
samples with a small clean teacher, and then purifies the model by knowledge distillation.

- **Paper:** `paper/adfl_eval (3).tex` (IEEEtran conference, 6-page limit)
- **Implementation:** `adfl-clean.ipynb` (single source of truth)
- **Extra-experiment launcher:** `run_extra_experiments.py` (thin wrapper: `run_extra.sh`)

## Layout

| Path | Contents |
|---|---|
| `adfl-clean.ipynb` | Full implementation, configuration, and experiment gating. Run cells top-to-bottom in one kernel. |
| `run_extra_experiments.py` / `run_extra.sh` | Parallel launcher for the extra experiments (runs 5–9). |
| `box-insurance/` | Base checkpoints (`.pt`), training histories (`.pkl`), and run-of-record JSON results. `comparison_tables.md` is the numeric source of truth. |
| `results_extra/` | Collected outputs of the extra experiments (JSON results + per-seed provenance checkpoints). |
| `paper/` | LaTeX source, `refs.bib`, and figures. |
| `EXTRA_RUNS.md` | Documents the extra experiments. |

## Requirements

Python 3 with `torch`, `torchvision`, `scikit-learn`, `seaborn`, `pandas`, `tqdm`, and
`matplotlib`. `papermill` is optional (the launcher falls back to `jupyter nbconvert`).

```bash
pip install -r requirements.txt
```

## Quick start (smoke test)

```bash
python run_extra_experiments.py --dry-run                    # print plan/env only
python run_extra_experiments.py --smoke --only exp1_partition_var
```

`--smoke` sets `ADFL_FIDELITY=fast`; leaving it unset runs the paper tier.

## Running the notebook directly

The notebook reads configuration from environment variables:

| Variable | Purpose |
|---|---|
| `ADFL_RUN` | Selects a preset run (unset defaults to RUN=2, Fashion-MNIST IID). |
| `ADFL_EXPERIMENT` | Gates the extra-experiment cells near the end of the notebook. |
| `ADFL_FIDELITY` | `fast` for the smoke tier; unset = paper tier. |
| `ADFL_WORK_DIR` | Where checkpoints are read/written. Always set this when running manually. |
| `ADFL_DATA_DIR` | Torchvision dataset cache (defaults to `./data`). |

## Extra experiments

```bash
python run_extra_experiments.py --only exp1_partition_var --jobs 1 --cores-per-run 8
```

`--only` takes one or more of the labels below. `--stage-from _empty --smoke --only ...`
forces a cold run with no staged checkpoints. CPU pinning is Linux-only (auto-detected and
skipped on Windows).

| Label | `ADFL_RUN` | `ADFL_EXPERIMENT` | Main output |
|---|---|---|---|
| `exp3_cifar_multiseed` | 7 | `""` | `results_cifar10_resnet18.json` |
| `exp1_partition_var` | 5 | `partition_var` | `partition_var_fashion_mnist_dirichlet0.5.json` |
| `exp2_teacher_gan_ablation` | 6 | `teacher_gan_ablation` | `teacher_gan_ablation_fashion_mnist_dirichlet0.5.json` |
| `exp4_mnist_noniid` | 8 | `mnist_noniid_control` | `results_mnist_dirichlet0.5.json` |
| `exp5_adaptive` | 9 | `adaptive_trigger` | `adaptive_trigger_fashion_mnist_dirichlet0.5.json` |

Paper-tier runs are GPU-scale and take hours (one RTX 3090 Ti: exp1 ~6.7 h, exp4 ~3.1 h,
exp5 ~21 h). Do not launch a paper-tier run without a plan for the time it will take.

## Reproducing the paper results

The JSON files in `box-insurance/` and `results_extra/`, together with
`box-insurance/comparison_tables.md`, are the numeric sources of truth. Reconcile any paper
numbers against these files, not against prose.

## Notes and gotchas

- Cells must run top-to-bottom in one kernel. Run one dataset per session; switching datasets
  mid-kernel leaves stale globals and mixed artifacts.
- Checkpoint resume is filename-based: names encode kind, dataset, model, attack, fidelity,
  partition, and seed. `FORCE_RETRAIN=True` rebuilds everything; `FORCE_REGEN_GAN=True` rebuilds
  only the GAN + distillation data.
- Keep `NUM_WORKERS=0`: notebook-defined classes cannot be unpickled in worker processes.
- Smoke and paper artifacts can coexist in `runs/*/work`. `--smoke` results are not evidence;
  delete `runs/` and `results_extra/` before a clean paper run.
- Before rerunning `exp5_adaptive`, delete `runs/exp5_adaptive/work/adaptive_*_FL*.pt`.

## Building the paper

`paper/adfl_eval (3).tex` is IEEEtran conference and requires `IEEEtran.cls`, which is **not**
included in this repository. Obtain it from CTAN or your TeX distribution before building.

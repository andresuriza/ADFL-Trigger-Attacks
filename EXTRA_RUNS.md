# Extra experiments (runs 5-7)

## What and why
- **Exp1 `exp1_partition_var` (ADFL_RUN=5)**: Fashion-MNIST Non-IID partition/teacher variance. The partition and teacher were fixed across seeds, so the Non-IID survival std was understated; this varies them.
- **Exp2 `exp2_teacher_gan_ablation` (ADFL_RUN=6)**: Fashion-MNIST Non-IID teacher/GAN 2x2 ablation, isolating why Non-IID hurts ADFL.
- **Exp3 `exp3_cifar_multiseed` (ADFL_RUN=7)**: CIFAR-10 IID headline over seeds 42,43,44, removing the single-seed limitation.

## Prerequisites
- Linux GPU box (one RTX 3090), Python 3, torch + torchvision + CUDA, `papermill` (falls back to `jupyter nbconvert`).
- `box-insurance/` checkpoints present in the repo root.

## Stage source (correction)
Base checkpoints are staged from **`box-insurance/`**, NOT from `vast-backup-20261015T204526Z-1-001.zip` (that zip holds only the MNIST run). Override with `--stage-from DIR`. Missing optional files only produce a warning.

## Commands
```bash
python run_extra_experiments.py --dry-run                    # show plan/env, do nothing
python run_extra_experiments.py --smoke                      # fast tier, whole pipeline
python run_extra_experiments.py --jobs 3 --cores-per-run 8   # full paper-tier run
# or: ./run_extra.sh --jobs 3 --cores-per-run 8
```
Use `--only LABEL` (repeatable) for a subset. Per-run files: `runs/<label>/{work/,log.txt,executed_<label>.ipynb}`. Exit code is non-zero if any run fails or yields no output JSON. Do not mix `--smoke` outputs with real ones: delete `runs/` and `results_extra/` between them.

## Outputs (`results_extra/`)
| File | Run | Paper use |
|---|---|---|
| `partition_var_fashion_mnist_dirichlet0.5.json` | Exp1 | Non-IID survival mean/std table (corrected variance) |
| `teacher_gan_ablation_fashion_mnist_dirichlet0.5.json` | Exp2 | Teacher x GAN 2x2 ablation table |
| `results_cifar10_resnet18.json` | Exp3 | CIFAR-10 headline table (`asr_post_mean/std`, `n_seeds=3`) |
| `backdoored_cifar10_resnet18_{pixel,semantic}_paper_iid_s43/_s44.pt` | Exp3 | Per-seed checkpoints (provenance) |

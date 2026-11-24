# Ataque a destilación adversaria ADFL mediante trigger indetectable

Proyecto de Aplicación de Ingenieria en Computadores.

## Descripción

Se contieene el código, checkpoints y resultados de los ataques y reimplementación de la defensa de ADFL (Zhu et al., 2023).

## Estructura

| Path | Contenidos |
|---|---|
| `adfl-clean.ipynb` | Implementación completa, configuración y experimentos. |
| `run_extra_experiments.py` / `run_extra.sh` | Launcher paralelo de experimentos adicionales (runs 5–9). |
| `box-insurance/` | Checkpoints (`.pt`), (`.pkl`), resultados JSON. |
| `results_extra/` | Resultados de experimentos extra |

## Requisitos

Python 3 con `torch`, `torchvision`, `scikit-learn`, `seaborn`, `pandas`, `tqdm`, `matplotlib`.

```bash
pip install -r requirements.txt
```

## Smoke test

```bash
python run_extra_experiments.py --dry-run
python run_extra_experiments.py --smoke --only exp1_partition_var
```

## Experimentos extra

```bash
python run_extra_experiments.py --only exp1_partition_var --jobs 1 --cores-per-run 8
```

`--only` toma las siguientes etiquetas. `--stage-from _empty --smoke --only ...`


| Etiqueta | `ADFL_RUN` | `ADFL_EXPERIMENT` | Salida |
|---|---|---|---|
| `exp3_cifar_multiseed` | 7 | `""` | `results_cifar10_resnet18.json` |
| `exp1_partition_var` | 5 | `partition_var` | `partition_var_fashion_mnist_dirichlet0.5.json` |
| `exp2_teacher_gan_ablation` | 6 | `teacher_gan_ablation` | `teacher_gan_ablation_fashion_mnist_dirichlet0.5.json` |
| `exp4_mnist_noniid` | 8 | `mnist_noniid_control` | `results_mnist_dirichlet0.5.json` |
| `exp5_adaptive` | 9 | `adaptive_trigger` | `adaptive_trigger_fashion_mnist_dirichlet0.5.json` |

Recomendado utilizar GPU.

## Reproducir resultados de paper

Los archivos JSON en `box-insurance/` y `results_extra/`, junto con
`box-insurance/comparison_tables.md`, son los resultados reales. Comparar números con estos archivos, además del paper.

## Credito

Desarrollado por Andres Uriza

# ADFL vulnerability — run-of-record comparison tables (paper fidelity, multi-seed)

Run of record: vast.ai RTX 3090 Ti, 2026-10-17, one consistent CUDA/cuDNN stack.
Grayscale (MNIST, Fashion): 100 clients / 25 malicious / 10 per round / 20 local epochs / 200 rounds, **3 seeds [42,43,44]**.
CIFAR-10 (ResNet18): 50 clients / 12 malicious / 10 per round / 10 local epochs / 100 rounds, **1 seed** (compute tier).

`excess` = (post-ADFL ASR) − teacher target rate T; "SURVIVES" when excess > 0.05, "WINNER" when > 0.15.
Grayscale base attacks **and** full-FL evasion confirmations are now **3-seed mean ± std** (ddof=1); excess from the mean.
MNIST and CIFAR evasion rows remain single-seed (MNIST = control, all removed; CIFAR = 1-seed tier).

---

## Table 1 — IID run of record (RUNs 1 / 2 / 4)

### Base attacks
| Dataset | Attack | ASR pre | ASR post | Clean ACC pre→post | T | Excess | Verdict | Seeds |
|---|---|---|---|---|---|---|---|---|
| MNIST | pixel | 1.000±0.000 | **0.002±0.002** | 0.989→0.945 | 0.001 | +0.001 | removed (control ✓) | 3 |
| Fashion | pixel | 1.000±0.000 | **0.145±0.170** | 0.907→0.746 | 0.000 | +0.145 | SURVIVES (very noisy) | 3 |
| Fashion | watermark | 0.993±0.003 | 0.001±0.001 | 0.907→0.741 | 0.000 | +0.000 | removed | 3 |
| CIFAR-10 | semantic | 0.241 | 0.034 | 0.648→0.147 | 0.034 | +0.000 | removed (ACC collapsed) | 1 |
| CIFAR-10 | pixel | 0.978 | 0.080 | 0.642→0.151 | 0.029 | +0.051 | SURVIVES (ACC collapsed) | 1 |

### Full-FL-confirmed evasion
| Dataset | Trigger | ASR post | Clean ACC pre→post | T | Excess | Verdict | Seeds |
|---|---|---|---|---|---|---|---|
| MNIST | three_corners | 0.000 | 0.990→0.946 | 0.003 | −0.003 | removed | 1 |
| MNIST | stripes_h | 0.078 | 0.990→0.947 | 0.000 | +0.078 | partial | 1 |
| MNIST | vertical_stripes | 0.030 | 0.991→0.944 | 0.019 | +0.011 | removed | 1 |
| Fashion | random_14×14_center | 0.024±0.040 | 0.911→0.735 | 0.000 | +0.024 | removed | 3 |
| Fashion | stripes_h | 0.009±0.006 | 0.913→0.746 | 0.000 | +0.009 | removed | 3 |
| Fashion | horizontal_stripes | 0.077±0.128 | 0.913→0.743 | 0.000 | +0.077 | partial (noisy) | 3 |
| CIFAR-10 | random_20×20_center | 0.505 | 0.636→0.182 | 0.000 | +0.505 | WINNER (ACC collapsed) | 1 |
| CIFAR-10 | vertical_stripes | 0.020 | 0.642→0.177 | 0.000 | +0.020 | removed | 1 |

**IID read:** under IID, ADFL mostly holds. The base pixel marginally survives but is wildly unstable (±0.170; run-to-run post-ADFL ASR swings 0.10–0.15). Every grayscale evasion trigger **collapses under 3 seeds** — the single-seed +0.095/+0.097 "survivors" were seed-42 artifacts. CIFAR survival sits on a model whose clean ACC collapsed to 0.15–0.18.

---

## Table 2 — Non-IID run of record (RUN 3, Dirichlet α = 0.5)

### Base attacks
| Dataset | Attack | ASR pre | ASR post | Clean ACC pre→post | T | Excess | Verdict | Seeds |
|---|---|---|---|---|---|---|---|---|
| Fashion | pixel | 1.000±0.000 | **0.152±0.136** | 0.890→0.652 | 0.005 | +0.147 | SURVIVES (noisy) | 3 |
| Fashion | watermark | 0.995±0.002 | **0.074±0.027** | 0.891→0.659 | 0.014 | +0.060 | SURVIVES (low variance) | 3 |

### Full-FL-confirmed evasion
| Dataset | Trigger | ASR post | Clean ACC pre→post | T | Excess | Verdict | Seeds |
|---|---|---|---|---|---|---|---|
| Fashion | **random_14×14_pixel** | **0.434±0.175** | 0.895→0.671 | 0.043 | **+0.390** | **WINNER (robust, n=3)** | 3 |
| Fashion | three_corners | 0.123±0.034 | 0.896→0.681 | 0.002 | +0.121 | partial (robust) | 3 |
| Fashion | random_20×20_pixel | 0.055±0.094 | 0.898→0.680 | 0.000 | +0.055 | partial (weak) | 3 |

---

## Headline findings (final, multi-seed)

1. **MNIST control passes** — ADFL removes the pixel backdoor (0.002 ± 0.002). Validates the reimplementation.
2. **ADFL holds under IID but fails under Non-IID** — the robustness opposite of the paper's claim, now with multi-seed support:
   - **Watermark:** fully removed under IID (0.001) but **survives under Non-IID (0.074 ± 0.027, +0.060)** — low variance, robust.
   - **Off-manifold evasion:** every IID evasion trigger collapses over 3 seeds, but under Non-IID **random_14×14_pixel survives at 0.434 ± 0.175 ASR (excess +0.390)** across 3 seeds — a robust Path-B winner. three_corners also holds a robust partial (+0.121 ± 0.034).
3. **CIFAR failure is GAN-distillation-DATA-quality, not off-manifold/structural** — ADFL collapses clean ACC to ~0.15; the real-data upper-bound gate removes the same backdoor cleanly at ~0.72 ACC.

## Caveats / limitations
- The grayscale **base-pixel** survival is high-variance and even run-to-run unstable (±0.14–0.17); the *robust* survival evidence is the Non-IID watermark (low variance) and the Non-IID random_14×14_pixel evasion winner.
- MNIST and CIFAR **evasion** rows are single-seed (MNIST control: all removed; CIFAR: 1-seed compute tier). The CIFAR +0.505 winner sits on a collapsed-ACC model, so the CIFAR argument rests on the real-data gate, not this ASR.
- `asr_post` in each results JSON is the seed-42 value; the 3-seed statistics are in `asr_post_mean` / `asr_post_std`.

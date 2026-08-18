# DASH: Dual-Branch Score Distillation for Guidance-Calibrated Compact Diffusion Models

Parameter compression of class-conditional diffusion models exposes a structural failure mode in existing distillation objectives: supervising only the conditional branch or the composite guided prediction leaves the classifier-free guidance gap underdetermined, admitting degenerate solutions where the student's guidance gap collapses at inference.

DASH resolves this through independent supervision of both score branches, together with TIRT Transfer — a mechanism that initialises the student's per-timestep training curriculum from the teacher's converged weights as a frozen prior.

---

## Requirements

```
pip install -r requirements.txt
```

Tested on Python 3.10, PyTorch 2.0.
CIFAR-10 / CIFAR-100 experiments run on a single NVIDIA T4 (16 GB).
ImageNet-64 experiments run on TPU.

---

## Repository structure

```
DASH/
├── models/unet.py              # ADM-style UNet (teacher 35.8M / student 6.1M)
├── diffusion/
│   ├── schedule.py             # Cosine noise schedule and forward process
│   ├── ddim.py                 # Deterministic DDIM sampler with CFG
│   └── cfg.py                  # Timestep-Adaptive Guidance (TAG) schedule
├── training/
│   ├── tirt.py                 # TIRT module and EMA helper
│   ├── losses.py               # L_im, L_un, L_an, L_DASH
│   └── checkpoint.py           # Checkpoint save / load / prune
├── configs/
│   ├── teacher_cifar10.yaml
│   ├── teacher_cifar100.yaml
│   ├── student_cifar10.yaml
│   └── student_cifar100.yaml
├── train_teacher.py            # Teacher training entry point
├── train_student.py            # DASH distillation entry point
└── evaluate.py                 # FID, IS, rho, cos(Delta), Gap MSE
```

---

## Training

**Step 1 — Train teacher**

```bash
python train_teacher.py \
    --config    configs/teacher_cifar10.yaml \
    --data_root /path/to/cifar \
    --out_dir   checkpoints/teacher_cifar10 \
    --download
```

**Step 2 — Distil student**

```bash
python train_student.py \
    --config       configs/student_cifar10.yaml \
    --data_root    /path/to/cifar \
    --teacher_ckpt checkpoints/teacher_cifar10/ckpt_epoch_0500.pth \
    --out_dir      checkpoints/student_cifar10
```

For CIFAR-100, substitute `cifar10` → `cifar100` throughout.

---

## Evaluation

```bash
python evaluate.py \
    --config       configs/student_cifar10.yaml \
    --ckpt         checkpoints/student_cifar10/ckpt_epoch_0300.pth \
    --teacher_ckpt checkpoints/teacher_cifar10/ckpt_epoch_0500.pth \
    --data_root    /path/to/cifar \
    --out_dir      eval_results/student_cifar10 \
    --seeds        42 123 456
```

Outputs: FID, IS (mean ± std over seeds), guidance gap ratio ρ, directional cosine cos(Δ), Gap MSE, and a `results.json` summary.

---

## Key hyperparameters

All paper hyperparameters are in `configs/`. Key values:

| Setting | Teacher | Student |
|---|---|---|
| Architecture | ch=128, nrb=2, 35.8M | ch=64, nrb=1, 6.1M |
| Compression | — | 5.9× |
| Iterations | 391K (~500 ep.) | 234K (~300 ep.) |
| Batch size | 64 | 64 |
| Learning rate | 2e-4 | 1e-4 |
| EMA decay | 0.9999 | 0.9990 |
| CFG dropout | 10% | dual-pass (none) |
| λ_im / λ_un / λ_an | — | 1.0 / 1.0 / 0.1 |
| TIRT γ | 5.0 | frozen from teacher |
| TAG w_min / w_max / β | 1.0 / 4.0 / 5.0 | — |
| DDIM steps (eval) | 50 | 50 |
| Seeds | 42, 123, 456 | 42, 123, 456 |

**ImageNet-64** uses the same loss weights and TIRT settings, with a 116.5M
teacher and a 22.3M student (5.2× compression), trained for 130 epochs at
global batch 2048 with cosine decay from 2e-4 and no warmup.

---

## Expected results

### CIFAR-10 / CIFAR-100 (mean ± std over 3 seeds)

| Model | Params | C10 FID ↓ | C10 IS ↑ | C100 FID ↓ | C100 IS ↑ | ρ (C10/C100) |
|---|---|---|---|---|---|---|
| Teacher | 35.8M | 5.47 ± 0.04 | 9.42 ± 0.05 | 6.80 ± 0.05 | 7.46 ± 0.04 | 1.00 |
| Scratch | 6.1M | 20.47 ± 0.31 | 7.44 ± 0.09 | 26.14 ± 0.44 | 6.51 ± 0.08 | — |
| Cond-only | 6.1M | 22.31 ± 0.46 | 7.21 ± 0.11 | 27.83 ± 0.59 | 5.98 ± 0.10 | 0.09 / 0.08 |
| Composite | 6.1M | 13.84 ± 0.22 | 8.63 ± 0.08 | 20.84 ± 0.37 | 6.84 ± 0.07 | 0.68 / 0.65 |
| Multi-*w* composite | 6.1M | 13.67 ± 0.19 | 8.96 ± 0.09 | 19.13 ± 0.37 | 7.23 ± 0.05 | 0.75 / 0.71 |
| FitNets | 6.1M | 12.84 ± 0.21 | 8.74 ± 0.08 | 19.47 ± 0.35 | 6.89 ± 0.07 | 0.63 / 0.61 |
| Gap Δ match | 6.1M | 11.42 ± 0.18 | 8.34 ± 0.07 | 18.46 ± 0.33 | 6.58 ± 0.07 | 0.81 / 0.78 |
| Magnitude pruning | 8.95M | 18.04 ± 0.33 | 8.56 ± 0.07 | 24.48 ± 0.43 | 6.89 ± 0.08 | 0.71 / 0.66 |
| Single-head (no-CFG) | 8.99M | 16.78 ± 0.25 | 9.00 ± 0.06 | 23.50 ± 0.47 | 7.19 ± 0.09 | — |
| **DASH (ours)** | **6.1M** | **8.87 ± 0.12** | **9.31 ± 0.06** | **10.47 ± 0.16** | **7.41 ± 0.05** | **0.91 / 0.89** |

### ImageNet-64 (1000 classes, 64×64)

| Model | Params | FID ↓ | IS ↑ | ρ | cos(Δ) | MSE_Δ |
|---|---|---|---|---|---|---|
| Teacher | 116.5M | 6.34 | 48.35 | 1.00 | 1.00 | 0.000 |
| Composite | 22.3M | 18.43 | 43.67 | 0.65 | 0.68 | 0.130 |
| w/o L_un | 22.3M | 16.95 | 44.53 | 0.61 | 0.72 | 0.153 |
| **DASH (ours)** | **22.3M** | **9.10** | **47.32** | **0.89** | **0.92** | **0.036** |

All results: 50-step DDIM, *w* = 4.0, 50K samples.

### Efficiency (T4, batch 64)

| Metric | Teacher | Student | Ratio |
|---|---|---|---|
| Parameters | 35.75M | 6.08M | 5.9× |
| FLOPs, 50-step CFG | 1247.8 G | 216.3 G | 5.77× |
| Latency | 14217.3 ms | 3925.8 ms | 3.62× |
| Peak memory | 821.1 MB | 479.9 MB | 1.71× |

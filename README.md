# DASH: Dual-Branch Score Distillation for Guidance-Calibrated Compact Diffusion Models

Parameter compression of class-conditional diffusion models exposes a structural failure mode in existing distillation objectives: supervising only the conditional branch or the composite guided prediction leaves the classifier-free guidance gap underdetermined, admitting degenerate solutions where the student's guidance gap collapses at inference.

DASH resolves this through independent supervision of both score branches, together with TIRT Transfer — a mechanism that initialises the student's per-timestep training curriculum from the teacher's converged weights as a frozen prior.

---

## Requirements

```
pip install -r requirements.txt
```

Tested on Python 3.10, PyTorch 2.0, single NVIDIA P100 (16 GB).

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
    --ckpt         checkpoints/student_cifar10/ckpt_epoch_0270.pth \
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
| Iterations | 391K (~500 ep.) | 234K (~270 ep.) |
| Batch size | 64 | 64 |
| Learning rate | 2e-4 | 2e-4 |
| EMA decay | 0.9999 | 0.9995 |
| λ_im / λ_un / λ_an | — | 1.0 / 1.0 / 0.1 |
| TIRT γ | 5.0 | frozen from teacher |
| TAG w_min / w_max | 1.0 / 4.0 | — |
| DDIM steps (eval) | 50 | 50 |
| Seeds | 42, 123, 456 | 42, 123, 456 |

---

## Expected results

| Model | CIFAR-10 FID ↓ | CIFAR-10 IS ↑ | CIFAR-100 FID ↓ | ρ |
|---|---|---|---|---|
| Teacher (35.8M) | 5.47 | 9.42 | 6.80 | 1.00 |
| DASH student (6.1M) | 8.87 | 9.31 | 10.47 | 0.91 |
| Composite baseline | 13.84 | 8.63 | 20.84 | 0.68 |
| Cond-only baseline | 22.31 | 7.21 | 27.83 | 0.09 |

50-step DDIM, w=4.0, 50K samples, mean over 3 seeds.

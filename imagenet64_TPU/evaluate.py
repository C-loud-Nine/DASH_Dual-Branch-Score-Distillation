#!/usr/bin/env python3
# =============================================================================
# evaluate.py — FID / IS and the calibration triple for ImageNet-64.
#
#   python evaluate.py --ckpt CKPT --data_root DIR --teacher_ckpt CKPT
#
# Sampling uses the TAG schedule (w_min=1.0, w_max=4.0, beta=5.0) unless
# --fixed_w is passed. This is the protocol reported in Table 2 and Supp. Tables D10/D12:
# teacher and student are evaluated with the identical sampler.
# =============================================================================

import argparse
import json
import os

import torch
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image

from config import load
from data import ImageNet64
from diffusion import cosine_alphas_cumprod, ddim_sample, q_sample
from unet import build


def load_ema(model, path):
    """Load a checkpoint and apply its EMA weights (Supp. Table A4)."""
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model_state_dict"])
    shadow = ck["ema_state_dict"]["shadow"]
    for n, p in model.named_parameters():
        if n in shadow:
            p.data.copy_(shadow[n])
    return model


@torch.no_grad()
def generate(model, cfg, abar, n, out_dir, device, batch=250,
             use_tag=True, w_max=4.0, seed=0, steps=50):
    os.makedirs(out_dir, exist_ok=True)
    if os.listdir(out_dir):
        raise RuntimeError(f"{out_dir} is not empty; use a new --out_dir")
    torch.manual_seed(seed)           # seeds the initial diffusion noise
    g = torch.Generator(device="cpu").manual_seed(seed)
    done = 0
    while done < n:
        b = min(batch, n - done)
        y = torch.randint(0, cfg.num_classes, (b,), generator=g).to(device)
        x = ddim_sample(model, (b, 3, cfg.image_size, cfg.image_size), y,
                        abar, cfg, steps, w_max, use_tag, device)
        for i in range(b):
            save_image((x[i] + 1) / 2, os.path.join(out_dir, f"{done+i:06d}.png"))
        done += b


@torch.no_grad()
def calibration(student, teacher, ds, abar, cfg, device, n_pairs=10_000):
    """rho, cos(Delta), MSE_Delta over matched teacher-student pairs."""
    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False)
    sum_s = sum_t = sum_cos = sum_mse = 0.0
    seen = 0
    for x0, y in loader:
        if seen >= n_pairs:
            break
        x0, y = x0[:n_pairs - seen].to(device), y[:n_pairs - seen].to(device)
        t = torch.randint(0, cfg.timesteps, (x0.shape[0],), device=device)
        xt = q_sample(x0, t, torch.randn_like(x0), abar)
        null = torch.full_like(y, cfg.num_classes)
        ds_ = (student(xt, t, y, force_class=True)
               - student(xt, t, null, force_class=True)).flatten(1).double()
        dt_ = (teacher(xt, t, y, force_class=True)
               - teacher(xt, t, null, force_class=True)).flatten(1).double()
        ns, nt = ds_.norm(dim=1), dt_.norm(dim=1)
        sum_s += ns.sum().item()
        sum_t += nt.sum().item()
        sum_cos += ((ds_ * dt_).sum(1) / (ns * nt).clamp(min=1e-8)).sum().item()
        sum_mse += (ds_ - dt_).pow(2).mean(1).sum().item()
        seen += x0.shape[0]
    if seen != n_pairs or sum_t <= 0:
        raise ValueError("insufficient calibration pairs or zero teacher gap")
    return {"rho": sum_s / sum_t, "cos_delta": sum_cos / seen,
            "mse_delta": sum_mse / seen}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/student.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--teacher_ckpt", default=None,
                   help="required for the calibration triple")
    p.add_argument("--is_teacher", action="store_true")
    p.add_argument("--fixed_w", action="store_true",
                   help="constant w instead of the TAG schedule")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="eval")
    a = p.parse_args()

    scfg = load(a.config)
    tcfg = scfg.teacher
    ecfg = scfg.eval
    if a.n is None:
        a.n = ecfg.n_samples
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = tcfg if a.is_teacher else scfg
    abar = cosine_alphas_cumprod(cfg.timesteps, device=device)

    model = build(cfg).to(device)
    load_ema(model, a.ckpt)
    model.eval()

    tag = f"{'fixedw' if a.fixed_w else 'tag'}_seed{a.seed}"
    gen = os.path.join(a.out_dir, f"gen_{tag}")
    ref = os.path.join(a.out_dir, "ref")
    generate(model, cfg, abar, a.n, gen, device,
             use_tag=not a.fixed_w, w_max=scfg.tag_w_max, seed=a.seed,
             steps=ecfg.ddim_steps)

    ds = ImageNet64(a.data_root, train=True)
    if len(os.listdir(ref) if os.path.isdir(ref) else []) < a.n:
        os.makedirs(ref, exist_ok=True)
        idx = torch.randperm(len(ds),
                             generator=torch.Generator().manual_seed(0))[:a.n]
        for j, i in enumerate(idx.tolist()):
            save_image((ds[i][0] + 1) / 2, os.path.join(ref, f"{j:06d}.png"))

    m = calculate_metrics(input1=gen, input2=ref, fid=True, isc=True,
                          cuda=torch.cuda.is_available(), verbose=False)
    out = {"fid": m["frechet_inception_distance"],
           "is": m["inception_score_mean"],
           "protocol": "fixed_w" if a.fixed_w else f"TAG w_max={scfg.tag_w_max}",
           "steps": ecfg.ddim_steps, "n": a.n, "seed": a.seed}

    if a.teacher_ckpt and not a.is_teacher:
        teacher = build(tcfg).to(device)
        load_ema(teacher, a.teacher_ckpt)
        teacher.eval()
        torch.manual_seed(a.seed)     # seeds calibration timesteps and noise
        out.update(calibration(model, teacher, ds, abar, cfg, device,
                               ecfg.calibration_pairs))
        out["calibration_inputs"] = {"split": "train",
                                     "indices": [0, ecfg.calibration_pairs],
                                     "batch": 100, "seed": a.seed}

    print(json.dumps(out, indent=2))
    with open(os.path.join(a.out_dir, f"results_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# =============================================================================
# evaluate.py — FID / IS and the calibration triple for ImageNet-64.
#
#   python evaluate.py --ckpt CKPT --data_root DIR --teacher_ckpt CKPT
#
# Sampling uses the TAG schedule (w_min=1.0, w_max=4.0, beta=5.0) unless
# --fixed_w is passed. This is the protocol reported in Table 2 and Table D7:
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
from losses import calibration_metrics
from unet import build


@torch.no_grad()
def generate(model, cfg, abar, n, out_dir, device, batch=250,
             use_tag=True, w_max=4.0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    done = len(os.listdir(out_dir))
    while done < n:
        b = min(batch, n - done)
        y = torch.randint(0, cfg.num_classes, (b,), generator=g).to(device)
        x = ddim_sample(model, (b, 3, cfg.image_size, cfg.image_size), y,
                        abar, tcfg, ecfg.ddim_steps, w_max, use_tag, device)
        for i in range(b):
            save_image((x[i] + 1) / 2, os.path.join(out_dir, f"{done+i:06d}.png"))
        done += b


@torch.no_grad()
def calibration(student, teacher, ds, abar, cfg, device, n_pairs=10_000):
    """rho, cos(Delta), MSE_Delta over matched teacher-student pairs."""
    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False)
    acc, seen = {"rho": 0.0, "cos_delta": 0.0, "mse_delta": 0.0}, 0
    for x0, y in loader:
        if seen >= n_pairs:
            break
        x0, y = x0.to(device), y.to(device)
        t = torch.randint(0, cfg.timesteps, (x0.shape[0],), device=device)
        xt = q_sample(x0, t, torch.randn_like(x0), abar)
        null = torch.full_like(y, cfg.num_classes)
        m = calibration_metrics(
            student(xt, t, y, force_class=True),
            student(xt, t, null, force_class=True),
            teacher(xt, t, y, force_class=True),
            teacher(xt, t, null, force_class=True))
        for k in acc:
            acc[k] += m[k] * x0.shape[0]
        seen += x0.shape[0]
    return {k: v / seen for k, v in acc.items()}


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
    p.add_argument("--n", type=int, default=ecfg.n_samples)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="eval")
    a = p.parse_args()

    scfg = load(a.config)
    tcfg = scfg.teacher
    ecfg = scfg.eval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = tcfg if a.is_teacher else scfg
    abar = cosine_alphas_cumprod(cfg.timesteps, device=device)

    model = build(cfg).to(device)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model_state_dict"])
    model.eval()

    gen = os.path.join(a.out_dir, "gen")
    ref = os.path.join(a.out_dir, "ref")
    generate(model, cfg, abar, a.n, gen, device,
             use_tag=not a.fixed_w, w_max=ecfg.w_max, seed=a.seed)

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
           "protocol": "fixed_w" if a.fixed_w else f"TAG w_max={ecfg.w_max}",
           "steps": ecfg.ddim_steps, "n": a.n, "seed": a.seed}

    if a.teacher_ckpt and not a.is_teacher:
        teacher = build(tcfg).to(device)
        teacher.load_state_dict(
            torch.load(a.teacher_ckpt, map_location="cpu")["model_state_dict"])
        teacher.eval()
        out.update(calibration(model, teacher, ds, abar, cfg, device,
                               ecfg.calibration_pairs))

    print(json.dumps(out, indent=2))
    with open(os.path.join(a.out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# =============================================================================
# train_teacher.py — ImageNet-64 teacher, TPU (torch_xla).
#
#   python train_teacher.py --data_root /path/to/imagenet64 --out_dir ckpt/teacher
#
# 130 epochs, global batch 2048, cosine LR from 2e-4 with no warmup, EMA 0.9999.
# The loss is TIRT-weighted (Eq. 2); TAG is inference-only and is not used here.
# =============================================================================

import argparse
import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from config import load
from data import ImageNet64
from diffusion import cosine_alphas_cumprod, q_sample
from tirt import TIRT, EMAHelper
from unet import build


def _run(rank, args):
    tcfg = load(args.config)
    device = xm.xla_device()
    torch.manual_seed(args.seed)

    ds = ImageNet64(args.data_root, train=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(),
        shuffle=True, seed=args.seed)
    per_core = tcfg.global_batch // xm.xrt_world_size()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=per_core, sampler=sampler, num_workers=4,
        drop_last=True)

    abar = cosine_alphas_cumprod(tcfg.timesteps, device=device)
    model = build(tcfg).to(device)
    tirt = TIRT(tcfg.timesteps, tcfg.tirt_snr_gamma,
                weight_objective=args.weight_objective,
                alphas_cumprod=abar.cpu()).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(tirt.parameters()),
        lr=tcfg.lr, betas=tcfg.adam_betas)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=tcfg.epochs)                              # no warmup
    ema = EMAHelper(mu=tcfg.ema_decay)
    ema.register(model)

    for epoch in range(tcfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        for x0, y in pl.MpDeviceLoader(loader, device):
            t = torch.randint(0, tcfg.timesteps, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            xt = q_sample(x0, t, eps, abar)

            pred = model(xt, t, y, cfg_dropout=tcfg.cfg_dropout)
            per = (pred - eps).pow(2).flatten(1).sum(1)
            loss = tirt(per, t, abar)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            xm.optimizer_step(opt)
            ema.update(model)
        sched.step()

        if xm.is_master_ordinal():
            xm.master_print(f"epoch {epoch+1}/{tcfg.epochs}  loss {loss.item():.4f}")
        if (epoch + 1) % 10 == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            xm.save({"model_state_dict": model.state_dict(),
                     "ema_state_dict": ema.state_dict(),
                     "tirt_state_dict": tirt.state_dict(),
                     "epoch": epoch + 1},
                    os.path.join(args.out_dir, f"teacher_ep{epoch+1:04d}.pth"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/teacher.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="ckpt/teacher")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_objective", default="descent",
                   choices=["descent", "ascent", "frozen"],
                   help="direction of the lambda update; see tirt.py")
    args = p.parse_args()
    xmp.spawn(_run, args=(args,), nprocs=None)

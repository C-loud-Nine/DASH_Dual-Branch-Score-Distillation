#!/usr/bin/env python3
# =============================================================================
# train_student.py — DASH distillation on ImageNet-64, TPU (torch_xla).
#
#   python train_student.py --data_root DIR --teacher_ckpt CKPT --out_dir DIR
#   python train_student.py ... --objective composite     # Table 2 baseline
#   python train_student.py ... --objective no_lun        # Table 2 ablation
#
# Both teacher and student run two forward passes per step (label y forced,
# then the null token), so eps^y and eps^0 are available separately and
# L_im / L_un act as independent constraints on each branch.
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
from losses import composite_loss, dash_losses
from tirt import TIRT, EMAHelper
from unet import build


def _run(rank, args):
    scfg = load(args.config)
    tcfg = scfg.teacher
    device = xm.xla_device()
    torch.manual_seed(args.seed)

    ds = ImageNet64(args.data_root, train=True)
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(),
        shuffle=True, seed=args.seed)
    per_core = scfg.global_batch // xm.xrt_world_size()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=per_core, sampler=sampler, num_workers=4, drop_last=True)

    abar = cosine_alphas_cumprod(scfg.timesteps, device=device)

    # ---- frozen teacher (EMA weights) ----
    ck = torch.load(args.teacher_ckpt, map_location="cpu")
    teacher = build(tcfg).to(device)
    teacher.load_state_dict(ck["model_state_dict"])
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)

    # ---- student ----
    student = build(scfg).to(device)

    # TIRT Transfer: copy the teacher's converged lambda and freeze it.
    # Under L_DASH the gradient on lambda measures student-teacher mismatch at
    # a converged teacher, not content difficulty, so it is held fixed.
    tirt = TIRT(scfg.timesteps, scfg.tirt_snr_gamma,
                weight_objective="frozen", alphas_cumprod=abar.cpu()).to(device)
    tirt.load_transferred(ck["tirt_state_dict"]["importance_adjustment"],
                          freeze=True)

    opt = torch.optim.AdamW(student.parameters(), lr=scfg.lr,
                            betas=scfg.adam_betas)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=scfg.epochs)
    ema = EMAHelper(mu=0.9995)
    ema.register(student)

    null_id = scfg.num_classes

    for epoch in range(scfg.epochs):
        student.train()
        sampler.set_epoch(epoch)
        for x0, y in pl.MpDeviceLoader(loader, device):
            t = torch.randint(0, scfg.timesteps, (x0.shape[0],), device=device)
            eps = torch.randn_like(x0)
            xt = q_sample(x0, t, eps, abar)
            null = torch.full_like(y, null_id)

            with torch.no_grad():
                e_y_t = teacher(xt, t, y, force_class=True)
                e_u_t = teacher(xt, t, null, force_class=True)

            e_y_s = student(xt, t, y, force_class=True)
            e_u_s = student(xt, t, null, force_class=True)

            w = tirt.get_weights(t, abar)
            l_im, l_un, l_an = dash_losses(e_y_s, e_u_s, e_y_t, e_u_t, eps, w)

            if args.objective == "dash":
                loss = (scfg.lambda_im * l_im + scfg.lambda_un * l_un
                        + scfg.lambda_an * l_an)
            elif args.objective == "no_lun":                 # ablation
                loss = scfg.lambda_im * l_im + scfg.lambda_an * l_an
            elif args.objective == "composite":              # baseline
                loss = composite_loss(e_y_s, e_u_s, e_y_t, e_u_t, w,
                                      w=tcfg.tag_w_max)
            else:
                raise ValueError(args.objective)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), scfg.grad_clip)
            xm.optimizer_step(opt)
            ema.update(student)
        sched.step()

        if xm.is_master_ordinal():
            xm.master_print(f"epoch {epoch+1}/{scfg.epochs}  loss {loss.item():.4f}")
        if (epoch + 1) % 10 == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            xm.save({"model_state_dict": student.state_dict(),
                     "ema_state_dict": ema.state_dict(),
                     "objective": args.objective,
                     "epoch": epoch + 1},
                    os.path.join(args.out_dir,
                                 f"{args.objective}_ep{epoch+1:04d}.pth"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/student.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--out_dir", default="ckpt/student")
    p.add_argument("--objective", default="dash",
                   choices=["dash", "no_lun", "composite"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    xmp.spawn(_run, args=(args,), nprocs=None)

# =============================================================================
# train_student.py
# DASH student distillation: dual-branch score distillation with TIRT Transfer.
#
# Usage:
#   python train_student.py \
#       --config       configs/student_cifar10.yaml \
#       --data_root    /path/to/cifar \
#       --teacher_ckpt /path/to/teacher_checkpoint.pth \
#       --out_dir      ./checkpoints/student_cifar10
#
#   # Resume from a previous student checkpoint:
#   python train_student.py \
#       --config       configs/student_cifar10.yaml \
#       --data_root    /path/to/cifar \
#       --teacher_ckpt /path/to/teacher_checkpoint.pth \
#       --out_dir      ./checkpoints/student_cifar10 \
#       --resume       ./checkpoints/student_cifar10/ckpt_epoch_0200.pth
#
# Distillation procedure:
#   1. Load teacher EMA weights from checkpoint and freeze all parameters.
#   2. Transfer teacher TIRT curriculum to student and freeze.
#   3. Train student with DASH objective (L_im + L_un + L_an).
#   4. Log per-component losses and guidance gap ratio rho at each val step.
# =============================================================================

import os
import argparse
import yaml
import types

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from models.unet import UNet
from diffusion.schedule import DiffusionSchedule
from diffusion.ddim import ddim_sample
from training.tirt import TIRT, EMAHelper
from training.losses import dash_loss
from training.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Config loader (shared with train_teacher.py)
# ---------------------------------------------------------------------------

def load_config(path: str):
    with open(path) as f:
        raw = yaml.safe_load(f)

    def to_ns(d):
        if isinstance(d, dict):
            ns = types.SimpleNamespace()
            for k, v in d.items():
                setattr(ns, k, to_ns(v))
            return ns
        if isinstance(d, list):
            return tuple(d)
        return d

    return to_ns(raw)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_loaders(cfg, data_root: str, download: bool = False):
    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    DS         = CIFAR10 if cfg.dataset == "cifar10" else CIFAR100
    full_train = DS(data_root, train=True,  download=download, transform=tfm)
    test_ds    = DS(data_root, train=False, download=download, transform=tfm)

    train_size = int(0.9 * len(full_train))
    val_size   = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    kw = dict(pin_memory=True, num_workers=4,
              persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, **kw)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Teacher loading
# ---------------------------------------------------------------------------

def load_teacher_frozen(teacher_model: UNet, ckpt_path: str,
                        ema_decay: float, device):
    """
    Load teacher EMA weights from checkpoint and freeze all parameters.
    Returns the teacher TIRT state dict (for TIRT Transfer to student).
    """
    print(f"Loading teacher checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    teacher_model.load_state_dict(ckpt["model_state_dict"])

    # Apply EMA weights
    ema_h = EMAHelper(ema_decay)
    ema_h.register(teacher_model)
    ema_h.load_state_dict(ckpt["ema_state_dict"])
    with torch.no_grad():
        for n, p in teacher_model.named_parameters():
            if p.requires_grad and n in ema_h.shadow:
                sd = ema_h.shadow[n]
                if sd.device != p.device:
                    sd = sd.to(p.device)
                p.data.copy_(sd)

    # Freeze teacher
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    teacher_model.eval()

    teacher_epoch = ckpt.get("epoch", "?")
    print(f"Teacher EMA loaded (epoch {teacher_epoch}), frozen.")
    return ckpt.get("tirt_state_dict", None)


# ---------------------------------------------------------------------------
# Guidance gap diagnostic (rho)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_guidance_gap_ratio(student_model, teacher_model, schedule,
                               val_loader, num_classes: int,
                               device, n_samples: int = 32):
    """
    Compute rho = E[||Delta_S||] / E[||Delta_T||] over a small validation batch.
    rho ~= 1.0 is ideal; rho -> 0 indicates guidance collapse.
    """
    student_model.eval()
    x0, cls = next(iter(val_loader))
    x0  = x0[:n_samples].to(device)
    cls = cls[:n_samples].to(device)

    t    = torch.randint(0, schedule.timesteps, (n_samples,), device=device).long()
    x_t  = schedule.q_sample(x0, t)
    null = torch.full((n_samples,), num_classes, device=device, dtype=torch.long)

    eps_S_c = student_model(x_t, t, cls, force_class=True)
    eps_S_u = student_model(x_t, t, null)
    eps_T_c = teacher_model(x_t, t, cls)
    eps_T_u = teacher_model(x_t, t, null)

    gap_S = (eps_S_c - eps_S_u).reshape(n_samples, -1).norm(dim=1).mean().item()
    gap_T = (eps_T_c - eps_T_u).reshape(n_samples, -1).norm(dim=1).mean().item()
    rho   = gap_S / (gap_T + 1e-8)

    student_model.train()
    return rho, gap_S, gap_T


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "samples"), exist_ok=True)

    # ------------------------------------------------------------------
    # Diffusion schedule
    # ------------------------------------------------------------------
    schedule = DiffusionSchedule(cfg.timesteps, device=str(device))

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_loader, val_loader = build_loaders(
        cfg, args.data_root, download=args.download)

    # ------------------------------------------------------------------
    # Teacher (frozen)
    # ------------------------------------------------------------------
    # Build teacher config namespace from student config's teacher_model block
    t_cfg = types.SimpleNamespace(
        image_size  = cfg.image_size,
        num_classes = cfg.num_classes,
        cfg_dropout = cfg.cfg_dropout,
        model       = cfg.teacher_model,
    )
    teacher_model = UNet(t_cfg).to(device)

    teacher_tirt_sd = load_teacher_frozen(
        teacher_model,
        ckpt_path  = args.teacher_ckpt,
        ema_decay  = 0.9999,
        device     = device,
    )

    t_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"Teacher: {t_params:,} params")

    # ------------------------------------------------------------------
    # Student
    # ------------------------------------------------------------------
    s_cfg = types.SimpleNamespace(
        image_size  = cfg.image_size,
        num_classes = cfg.num_classes,
        cfg_dropout = cfg.cfg_dropout,
        model       = cfg.student_model,
    )
    student_model = UNet(s_cfg).to(device)
    s_params = sum(p.numel() for p in student_model.parameters())
    print(f"Student: {s_params:,} params  "
          f"({t_params/s_params:.1f}x compression)")

    # ------------------------------------------------------------------
    # TIRT Transfer: copy teacher curriculum -> student, then FREEZE
    # ------------------------------------------------------------------
    student_tirt = TIRT(cfg.timesteps, cfg.tirt_snr_gamma).to(device)
    if teacher_tirt_sd is not None:
        student_tirt.load_state_dict(teacher_tirt_sd)
        print("TIRT curriculum transferred from teacher.")
    else:
        print("Warning: no teacher TIRT found — using random init.")
    for p in student_tirt.parameters():
        p.requires_grad_(False)
    student_tirt.eval()

    # ------------------------------------------------------------------
    # Optimiser, scheduler, EMA, scaler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(student_model.parameters(),
                            lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr_min)
    scaler      = GradScaler()
    student_ema = EMAHelper(cfg.ema_decay)
    student_ema.register(student_model)

    ckpt_mgr = CheckpointManager(args.out_dir, keep_n=cfg.keep_n)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    train_loss_hist = []
    val_loss_hist   = []
    L_im_hist = []
    L_un_hist = []
    L_an_hist = []

    if args.resume and os.path.exists(args.resume):
        sd = ckpt_mgr.load(args.resume, device=str(device))
        student_model.load_state_dict(sd["model_state_dict"])
        optimizer.load_state_dict(sd["optimizer_state_dict"])
        scaler.load_state_dict(sd["scaler_state_dict"])
        scheduler.load_state_dict(sd["scheduler_state_dict"])
        student_ema.load_state_dict(sd["ema_state_dict"])
        train_loss_hist = sd.get("train_loss_hist", [])
        val_loss_hist   = sd.get("val_loss_hist",   [])
        L_im_hist = sd.get("L_im_hist", [])
        L_un_hist = sd.get("L_un_hist", [])
        L_an_hist = sd.get("L_an_hist", [])
        start_epoch = sd["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*68}")
    print(f"  DASH distillation: {cfg.dataset.upper()}")
    print(f"  Epochs {start_epoch+1}–{cfg.num_epochs}  "
          f"batch={cfg.batch_size}  lr={cfg.lr}")
    print(f"  Loss: {cfg.lam_im}*L_im + {cfg.lam_un}*L_un + {cfg.lam_an}*L_an")
    print(f"  Student {s_params/1e6:.1f}M <- Teacher {t_params/1e6:.1f}M  "
          f"({t_params/s_params:.1f}x)")
    print(f"{'='*68}\n")

    for ep in range(start_epoch, cfg.num_epochs):
        epoch_num = ep + 1
        student_model.train()

        ep_total = ep_im = ep_un = ep_an = 0.0

        pbar = tqdm(train_loader,
                    desc=f"Ep {epoch_num:03d}/{cfg.num_epochs}", leave=True)

        for x0, cls in pbar:
            x0  = x0.to(device, non_blocking=True)
            cls = cls.to(device, non_blocking=True)
            B   = x0.shape[0]
            t   = torch.randint(0, cfg.timesteps, (B,), device=device).long()
            eps = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, eps)

            null_cls = torch.full((B,), cfg.num_classes,
                                  device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                # Teacher branch targets (no gradient)
                with torch.no_grad():
                    eps_T_cond   = teacher_model(x_t, t, cls)
                    eps_T_uncond = teacher_model(x_t, t, null_cls)

                # Student predictions
                # force_class=True: bypasses CFG dropout for conditional pass
                eps_S_cond   = student_model(x_t, t, cls,      force_class=True)
                eps_S_uncond = student_model(x_t, t, null_cls)

                # DASH objective: L_im + L_un + L_an
                loss, L_im, L_un, L_an = dash_loss(
                    eps_S_cond, eps_S_uncond,
                    eps_T_cond, eps_T_uncond,
                    eps, t,
                    student_tirt,
                    schedule.alphas_cumprod,
                    lam_im=cfg.lam_im,
                    lam_un=cfg.lam_un,
                    lam_an=cfg.lam_an,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            student_ema.update(student_model)

            ep_total += loss.item()
            ep_im    += L_im.item()
            ep_un    += L_un.item()
            ep_an    += L_an.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                Lim=f"{L_im.item():.3f}",
                Lun=f"{L_un.item():.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        n_b = len(train_loader)
        train_loss_hist.append(ep_total / n_b)
        L_im_hist.append(ep_im / n_b)
        L_un_hist.append(ep_un / n_b)
        L_an_hist.append(ep_an / n_b)
        scheduler.step()

        # Validation + guidance gap diagnostic
        do_val = (ep == 0) or (epoch_num % cfg.val_every == 0)
        if do_val:
            student_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x0v, clsv in val_loader:
                    x0v  = x0v.to(device)
                    clsv = clsv.to(device)
                    tv   = torch.randint(0, cfg.timesteps,
                                        (x0v.shape[0],), device=device).long()
                    ev   = torch.randn_like(x0v)
                    xv   = schedule.q_sample(x0v, tv, ev)
                    val_loss += F.mse_loss(
                        student_model(xv, tv, clsv), ev).item()
            val_loss /= len(val_loader)
            val_loss_hist.append(val_loss)

            # Guidance gap ratio rho
            rho, gap_s, gap_t = compute_guidance_gap_ratio(
                student_model, teacher_model, schedule,
                val_loader, cfg.num_classes, device)

            print(f"\n  Ep {epoch_num:03d} | "
                  f"train={ep_total/n_b:.4f}  val={val_loss:.4f} | "
                  f"Lim={ep_im/n_b:.4f}  Lun={ep_un/n_b:.4f}  "
                  f"Lan={ep_an/n_b:.5f} | "
                  f"gap_S={gap_s:.3f}  gap_T={gap_t:.3f}  rho={rho:.3f}")

            if rho < 0.5 and epoch_num > 30:
                print("  WARNING: rho < 0.5 — "
                      "check that L_un is contributing to the gradient.")

            student_model.train()

        # Checkpoint + sample
        if epoch_num % cfg.checkpoint_every == 0:
            try:
                ema_model = student_ema.ema_copy(student_model)
                ema_model.eval()
                with torch.no_grad():
                    cls_g = torch.arange(
                        min(cfg.num_classes, 10), device=device
                    ).repeat(2)[:16]
                    imgs = ddim_sample(
                        ema_model, schedule,
                        (16, 3, cfg.image_size, cfg.image_size),
                        num_classes=cfg.num_classes,
                        class_labels=cls_g,
                        ddim_steps=20,
                        w_eval=cfg.tag_w_max,
                        use_tag=True,
                        device=str(device),
                    )
                imgs = torch.clamp((imgs + 1) / 2, 0, 1)
                save_image(
                    imgs,
                    os.path.join(args.out_dir, "samples",
                                 f"epoch_{epoch_num:04d}.png"),
                    nrow=4,
                )
                del ema_model
            except Exception as e:
                print(f"  Sample generation failed: {e}")

            ckpt_mgr.save({
                "epoch":                epoch_num,
                "model_state_dict":     student_model.state_dict(),
                "ema_state_dict":       student_ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss_hist":      train_loss_hist,
                "val_loss_hist":        val_loss_hist,
                "L_im_hist":            L_im_hist,
                "L_un_hist":            L_un_hist,
                "L_an_hist":            L_an_hist,
            }, epoch_num)

    print("Distillation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DASH student distillation.")
    parser.add_argument("--config",       required=True,
                        help="Path to student YAML config.")
    parser.add_argument("--data_root",    required=True,
                        help="Root directory containing CIFAR dataset.")
    parser.add_argument("--teacher_ckpt", required=True,
                        help="Path to teacher checkpoint (.pth).")
    parser.add_argument("--out_dir",      required=True,
                        help="Directory to save student checkpoints.")
    parser.add_argument("--resume",       default=None,
                        help="Path to student checkpoint to resume from.")
    parser.add_argument("--download",     action="store_true",
                        help="Download dataset if not present.")
    args = parser.parse_args()
    train(args)

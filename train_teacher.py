# =============================================================================
# train_teacher.py
# Train the DASH teacher model with TIRT + TAG on CIFAR-10 or CIFAR-100.
#
# Usage:
#   python train_teacher.py --config configs/teacher_cifar10.yaml \
#                           --data_root /path/to/cifar \
#                           --out_dir   ./checkpoints/teacher_cifar10
#
#   python train_teacher.py --config configs/teacher_cifar100.yaml \
#                           --data_root /path/to/cifar \
#                           --out_dir   ./checkpoints/teacher_cifar100 \
#                           --resume    ./checkpoints/teacher_cifar100/ckpt_epoch_0200.pth
#
# The dataset directory should contain the standard torchvision CIFAR layout
# (downloaded automatically if absent when --download is passed).
# =============================================================================

import os
import math
import argparse
import yaml
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import save_image, make_grid
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from models.unet import UNet
from diffusion.schedule import DiffusionSchedule
from diffusion.ddim import ddim_sample
from training.tirt import TIRT, EMAHelper
from training.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Load YAML config and return a nested SimpleNamespace object."""
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
    """Build train / val / test DataLoaders for CIFAR-10 or CIFAR-100."""
    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    DS = CIFAR10 if cfg.dataset == "cifar10" else CIFAR100
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
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset: {cfg.dataset.upper()}  "
          f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    return train_loader, val_loader, test_loader


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
        print(f"GPU: {torch.cuda.get_device_name(0)}")
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
    train_loader, val_loader, _ = build_loaders(
        cfg, args.data_root, download=args.download)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = UNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher UNet: {n_params:,} params  ({n_params*4/1024**2:.0f} MB)")

    # ------------------------------------------------------------------
    # TIRT
    # ------------------------------------------------------------------
    tirt = TIRT(cfg.timesteps, cfg.tirt_snr_gamma).to(device)
    print("TIRT enabled.")

    # ------------------------------------------------------------------
    # Optimiser, scheduler, EMA, scaler
    # ------------------------------------------------------------------
    params    = list(model.parameters()) + list(tirt.parameters())
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs, eta_min=cfg.lr_min)
    scaler    = GradScaler()
    ema       = EMAHelper(cfg.ema_decay)
    ema.register(model)

    ckpt_mgr = CheckpointManager(args.out_dir, keep_n=cfg.keep_n)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch  = 0
    train_losses = []
    val_losses   = []

    if args.resume and os.path.exists(args.resume):
        sd = ckpt_mgr.load(args.resume, device=str(device))
        model.load_state_dict(sd["model_state_dict"])
        optimizer.load_state_dict(sd["optimizer_state_dict"])
        scaler.load_state_dict(sd["scaler_state_dict"])
        scheduler.load_state_dict(sd["scheduler_state_dict"])
        ema.load_state_dict(sd["ema_state_dict"])
        tirt.load_state_dict(sd["tirt_state_dict"])
        train_losses = sd.get("train_losses", [])
        val_losses   = sd.get("val_losses", [])
        start_epoch  = sd["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Teacher training: {cfg.dataset.upper()}")
    print(f"  Epochs {start_epoch+1}–{cfg.num_epochs}  "
          f"batch={cfg.batch_size}  lr={cfg.lr}")
    print(f"  TIRT gamma={cfg.tirt_snr_gamma}  "
          f"TAG w_max={cfg.tag_w_max}  EMA={cfg.ema_decay}")
    print(f"{'='*65}\n")

    for ep in range(start_epoch, cfg.num_epochs):
        epoch_num = ep + 1
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch_num:03d}/{cfg.num_epochs}", leave=True)
        for x0, cls in pbar:
            x0  = x0.to(device, non_blocking=True)
            cls = cls.to(device, non_blocking=True)
            B   = x0.shape[0]
            t   = torch.randint(0, cfg.timesteps, (B,), device=device).long()
            eps = torch.randn_like(x0)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                x_t       = schedule.q_sample(x0, t, eps)
                pred      = model(x_t, t, cls)
                loss_per  = F.mse_loss(pred, eps, reduction="none").view(B, -1).mean(1)
                loss      = tirt(loss_per, t, schedule.alphas_cumprod)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        # Validation
        if ep == 0 or epoch_num % cfg.val_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x0v, clsv in val_loader:
                    x0v  = x0v.to(device)
                    clsv = clsv.to(device)
                    tv   = torch.randint(0, cfg.timesteps,
                                        (x0v.shape[0],), device=device).long()
                    ev   = torch.randn_like(x0v)
                    xv   = schedule.q_sample(x0v, tv, ev)
                    val_loss += F.mse_loss(model(xv, tv, clsv), ev).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            print(f"\n  Ep {epoch_num:03d} | "
                  f"train={avg_loss:.4f}  val={val_loss:.4f}  "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
            model.train()

        # Checkpoint + sample
        if epoch_num % cfg.checkpoint_every == 0:
            # Quick sample with EMA model
            try:
                ema_model = ema.ema_copy(model)
                ema_model.eval()
                with torch.no_grad():
                    cls_g = torch.arange(
                        min(cfg.num_classes, 10), device=device
                    ).repeat(2)[:16]
                    imgs = ddim_sample(
                        ema_model, schedule, (16, 3, cfg.image_size, cfg.image_size),
                        num_classes=cfg.num_classes,
                        class_labels=cls_g,
                        ddim_steps=20,
                        w_eval=cfg.tag_w_max,
                        use_tag=True,
                        device=str(device),
                    )
                imgs = torch.clamp((imgs + 1) / 2, 0, 1)
                save_image(imgs,
                           os.path.join(args.out_dir, "samples",
                                        f"epoch_{epoch_num:04d}.png"),
                           nrow=4)
                del ema_model
            except Exception as e:
                print(f"  Sample generation failed: {e}")

            ckpt_mgr.save({
                "epoch":                epoch_num,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict":    scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict":       ema.state_dict(),
                "tirt_state_dict":      tirt.state_dict(),
                "train_losses":         train_losses,
                "val_losses":           val_losses,
            }, epoch_num)

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DASH teacher model.")
    parser.add_argument("--config",    required=True,
                        help="Path to YAML config (e.g. configs/teacher_cifar10.yaml).")
    parser.add_argument("--data_root", required=True,
                        help="Root directory containing CIFAR dataset.")
    parser.add_argument("--out_dir",   required=True,
                        help="Directory to save checkpoints and samples.")
    parser.add_argument("--resume",    default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--download",  action="store_true",
                        help="Download dataset if not present.")
    args = parser.parse_args()
    train(args)

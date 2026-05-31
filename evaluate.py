# =============================================================================
# evaluate.py
# Evaluate a trained student (or teacher) checkpoint.
# Computes: FID, IS, guidance gap ratio rho, directional cosine cos(Delta),
#           and gap MSE — the full calibration diagnostic suite from the paper.
#
# Usage:
#   # Evaluate student on CIFAR-10 (3 seeds, 50K samples):
#   python evaluate.py \
#       --config      configs/student_cifar10.yaml \
#       --ckpt        ./checkpoints/student_cifar10/ckpt_epoch_0270.pth \
#       --teacher_ckpt ./checkpoints/teacher_cifar10/ckpt_epoch_0500.pth \
#       --data_root   /path/to/cifar \
#       --out_dir     ./eval_results/student_cifar10 \
#       --seeds       42 123 456
#
#   # Evaluate teacher only (no calibration metrics):
#   python evaluate.py \
#       --config      configs/teacher_cifar10.yaml \
#       --ckpt        ./checkpoints/teacher_cifar10/ckpt_epoch_0500.pth \
#       --data_root   /path/to/cifar \
#       --out_dir     ./eval_results/teacher_cifar10 \
#       --teacher_only
#
# Calibration metrics (rho, cos(Delta), Gap MSE) require --teacher_ckpt
# to be provided so that matched teacher-student pairs can be computed.
# =============================================================================

import os
import json
import shutil
import argparse
import yaml
import types

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.unet import UNet
from diffusion.schedule import DiffusionSchedule
from diffusion.ddim import ddim_sample
from training.tirt import EMAHelper
from training.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Config loader
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
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_ema_model(model: UNet, ckpt_path: str,
                   ema_decay: float, device) -> UNet:
    """Load EMA weights from checkpoint into a copy of model."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    ema = EMAHelper(ema_decay)
    ema.register(model)
    ema.load_state_dict(ckpt["ema_state_dict"])

    # Apply EMA weights to model in-place
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and n in ema.shadow:
                sd = ema.shadow[n]
                if sd.device != p.device:
                    sd = sd.to(p.device)
                p.data.copy_(sd)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded EMA model from {ckpt_path} (epoch {epoch})")
    return model


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def generate_images(model, schedule, out_dir: str, n_samples: int,
                    num_classes: int, image_size: int, ddim_steps: int,
                    w_eval: float, use_tag: bool, tag_w_min: float,
                    tag_beta: float, batch_size: int, device) -> str:
    """
    Generate n_samples images and save as PNG files under out_dir.
    Returns out_dir path.
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    done = 0
    pbar = tqdm(total=n_samples, desc=f"Generating ({ddim_steps} steps)")

    while done < n_samples:
        bs  = min(batch_size, n_samples - done)
        cls = torch.randint(0, num_classes, (bs,), device=device)

        with torch.no_grad():
            imgs = ddim_sample(
                model, schedule,
                shape=(bs, 3, image_size, image_size),
                num_classes=num_classes,
                class_labels=cls,
                ddim_steps=ddim_steps,
                w_eval=w_eval,
                use_tag=use_tag,
                w_min=tag_w_min,
                tag_beta=tag_beta,
                device=str(device),
            )

        imgs = torch.clamp((imgs + 1) / 2, 0, 1)
        for i in range(bs):
            save_image(imgs[i], os.path.join(out_dir, f"{done:06d}.png"))
            done += 1
        pbar.update(bs)

    pbar.close()
    return out_dir


# ---------------------------------------------------------------------------
# FID + IS
# ---------------------------------------------------------------------------

def compute_fid_is(img_dir: str, dataset: str, device_str: str):
    """Compute FID and IS using torch-fidelity."""
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        raise ImportError("torch-fidelity not installed. "
                          "Run: pip install torch-fidelity")

    ref = "cifar10-train" if dataset == "cifar10" else None
    if ref is None:
        print("Warning: no built-in reference for CIFAR-100; "
              "FID requires a precomputed statistics file.")

    m = calculate_metrics(
        input1=img_dir,
        input2=ref,
        cuda=(device_str != "cpu"),
        isc=True,
        fid=True,
        kid=False,
        verbose=False,
    )
    return (m["frechet_inception_distance"],
            m["inception_score_mean"],
            m["inception_score_std"])


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_calibration_metrics(
    student_model,
    teacher_model,
    schedule,
    data_loader,
    num_classes: int,
    device,
    n_pairs: int = 10000,
):
    """
    Compute guidance-gap calibration metrics over matched teacher-student pairs:
        rho        = E[||Delta_S||] / E[||Delta_T||]   (magnitude ratio)
        cos(Delta) = E[Delta_S . Delta_T / (||Delta_S|| ||Delta_T||)]
        Gap MSE    = E[||Delta_S - Delta_T||^2]

    rho ~= 1 and cos(Delta) ~= 1 indicate ideal calibration.
    """
    student_model.eval()
    teacher_model.eval()

    norm_S_list  = []
    norm_T_list  = []
    cos_list     = []
    gap_mse_list = []
    collected    = 0

    pbar = tqdm(total=n_pairs, desc="Computing calibration metrics")

    while collected < n_pairs:
        for x0, cls in data_loader:
            if collected >= n_pairs:
                break
            x0  = x0.to(device)
            cls = cls.to(device)
            B   = x0.shape[0]
            t   = torch.randint(0, schedule.timesteps, (B,), device=device).long()
            x_t = schedule.q_sample(x0, t)
            null = torch.full((B,), num_classes, device=device, dtype=torch.long)

            eps_S_c = student_model(x_t, t, cls, force_class=True)
            eps_S_u = student_model(x_t, t, null)
            eps_T_c = teacher_model(x_t, t, cls)
            eps_T_u = teacher_model(x_t, t, null)

            delta_S = eps_S_c - eps_S_u  # (B, C, H, W)
            delta_T = eps_T_c - eps_T_u

            # Flatten per sample
            dS_flat = delta_S.reshape(B, -1)
            dT_flat = delta_T.reshape(B, -1)

            norm_S = dS_flat.norm(dim=1)          # (B,)
            norm_T = dT_flat.norm(dim=1)

            cos = (dS_flat * dT_flat).sum(dim=1) / (norm_S * norm_T + 1e-8)

            gap_mse = F.mse_loss(delta_S, delta_T, reduction="none") \
                        .reshape(B, -1).mean(dim=1)

            norm_S_list.extend(norm_S.cpu().tolist())
            norm_T_list.extend(norm_T.cpu().tolist())
            cos_list.extend(cos.cpu().tolist())
            gap_mse_list.extend(gap_mse.cpu().tolist())
            collected += B
            pbar.update(B)

    pbar.close()

    rho     = np.mean(norm_S_list) / (np.mean(norm_T_list) + 1e-8)
    cos_val = np.mean(cos_list)
    gap_mse = np.mean(gap_mse_list)

    return rho, cos_val, gap_mse


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    schedule = DiffusionSchedule(cfg.timesteps, device=str(device))

    # Determine if evaluating student or teacher
    is_student = hasattr(cfg, "student_model") and not args.teacher_only

    # Build model
    if is_student:
        model_cfg = types.SimpleNamespace(
            image_size=cfg.image_size, num_classes=cfg.num_classes,
            cfg_dropout=cfg.cfg_dropout, model=cfg.student_model)
        ema_decay = cfg.ema_decay
    else:
        model_cfg = cfg
        ema_decay = cfg.ema_decay

    eval_model = UNet(model_cfg).to(device)
    eval_model = load_ema_model(eval_model, args.ckpt, ema_decay, device)

    # Teacher model for calibration metrics
    teacher_model = None
    if is_student and args.teacher_ckpt:
        t_cfg = types.SimpleNamespace(
            image_size=cfg.image_size, num_classes=cfg.num_classes,
            cfg_dropout=cfg.cfg_dropout, model=cfg.teacher_model)
        teacher_model = UNet(t_cfg).to(device)
        teacher_model = load_ema_model(
            teacher_model, args.teacher_ckpt, 0.9999, device)

    # Dataset (for calibration metrics)
    tfm = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    DS = CIFAR10 if cfg.dataset == "cifar10" else CIFAR100
    test_ds = DS(args.data_root, train=False, download=args.download, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=cfg.fid_batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Generate images
        img_dir = os.path.join(args.out_dir, f"generated_seed{seed}")
        generate_images(
            eval_model, schedule, img_dir,
            n_samples   = cfg.fid_samples,
            num_classes = cfg.num_classes,
            image_size  = cfg.image_size,
            ddim_steps  = cfg.ddim_steps,
            w_eval      = cfg.tag_w_max,
            use_tag     = True,
            tag_w_min   = cfg.tag_w_min,
            tag_beta    = cfg.tag_beta,
            batch_size  = cfg.fid_batch_size,
            device      = device,
        )

        # FID + IS
        fid, is_mean, is_std = compute_fid_is(
            img_dir, cfg.dataset, str(device))
        print(f"  FID: {fid:.2f}  IS: {is_mean:.2f} ± {is_std:.2f}")

        seed_results = {"fid": fid, "is_mean": is_mean, "is_std": is_std}

        # Calibration metrics (student only)
        if is_student and teacher_model is not None:
            rho, cos_d, gap_mse = compute_calibration_metrics(
                eval_model, teacher_model, schedule,
                test_loader, cfg.num_classes, device,
                n_pairs=10000,
            )
            print(f"  rho={rho:.3f}  cos(Delta)={cos_d:.3f}  "
                  f"Gap MSE={gap_mse:.4f}")
            seed_results.update({
                "rho": rho, "cos_delta": cos_d, "gap_mse": gap_mse})

        results[f"seed_{seed}"] = seed_results

    # Summary (mean ± std over seeds)
    print(f"\n{'='*60}")
    print(f"  Summary over seeds {args.seeds}")
    print(f"{'='*60}")
    for metric in ["fid", "is_mean", "rho", "cos_delta", "gap_mse"]:
        vals = [r[metric] for r in results.values() if metric in r]
        if vals:
            print(f"  {metric:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Save results
    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DASH model.")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--ckpt",         required=True,
                        help="Checkpoint to evaluate.")
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--out_dir",      required=True)
    parser.add_argument("--teacher_ckpt", default=None,
                        help="Teacher checkpoint for calibration metrics.")
    parser.add_argument("--seeds",        nargs="+", type=int,
                        default=[42, 123, 456])
    parser.add_argument("--teacher_only", action="store_true",
                        help="Evaluate a teacher checkpoint (no calibration).")
    parser.add_argument("--download",     action="store_true")
    args = parser.parse_args()
    evaluate(args)

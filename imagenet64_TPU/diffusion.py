# =============================================================================
# diffusion.py — cosine schedule, DDIM sampler, and the TAG guidance schedule.
# =============================================================================

import math

import torch


# ----------------------------------------------------------------- schedule
def cosine_alphas_cumprod(T=1000, s=0.008, device="cpu"):
    """Nichol & Dhariwal cosine schedule."""
    x = torch.arange(T + 1, dtype=torch.float64, device=device)
    f = torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
    abar = (f / f[0])[:T]
    return abar.clamp(1e-8, 1.0 - 1e-4).float()


def q_sample(x0, t, eps, abar):
    """Forward process x_t = sqrt(abar) x0 + sqrt(1-abar) eps."""
    a = abar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1.0 - a).sqrt() * eps


# ---------------------------------------------------------------------- TAG
def tag_schedule(t_val, w_min, w_max, beta, T, device="cpu"):
    """Main paper Eq. 3.

    w(t) = w_min + (w_max - w_min) * sigmoid(beta (0.5 - t/T))

    Low guidance at high noise, high guidance at low noise. Applied at
    inference only; the branch targets eps^y_T and eps^0_T come from separate
    forward passes and are independent of w(t).
    """
    return w_min + (w_max - w_min) * torch.sigmoid(
        torch.tensor(beta * (0.5 - t_val / T), device=device)
    )


# -------------------------------------------------------------- DDIM + CFG
@torch.no_grad()
def ddim_sample(model, shape, class_labels, abar, cfg, ddim_steps=50,
                w_max=4.0, use_tag=True, device="cpu"):
    """Deterministic DDIM with classifier-free guidance.

    use_tag=True  -> w varies over t per Eq. 3 (the reported protocol)
    use_tag=False -> constant w = w_max (fixed-w ablation rows only)
    """
    T = cfg.timesteps
    x = torch.randn(shape, device=device)
    null = torch.full_like(class_labels, cfg.num_classes)   # null token
    step = max(T // ddim_steps, 1)
    ts = list(range(T - 1, -1, -step))[:ddim_steps]

    for i, t_val in enumerate(ts):
        t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

        w = (tag_schedule(t_val, cfg.tag_w_min, cfg.tag_w_max,
                          cfg.tag_beta, T, device) if use_tag
             else torch.tensor(float(w_max), device=device))

        e_c = model(x, t, class_labels, force_class=True)
        e_u = model(x, t, null, force_class=True)
        eps = e_u + w * (e_c - e_u)

        a_t = abar[t_val]
        a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps

    return x.clamp(-1, 1)

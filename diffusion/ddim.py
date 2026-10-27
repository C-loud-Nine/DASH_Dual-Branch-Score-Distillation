# =============================================================================
# diffusion/ddim.py
# Deterministic DDIM sampler (Song et al. 2020, eta=0).
#
# Used for both teacher evaluation (with TAG guidance schedule) and
# student evaluation (with fixed or TAG guidance).
# The sampler runs two forward passes per step — conditional and
# unconditional — and combines via CFG:
#   eps = eps_uncond + w(t) * (eps_cond - eps_uncond)
# =============================================================================

import torch
from .cfg import tag_guidance_weight


@torch.no_grad()
def ddim_sample(
    model,
    schedule,
    shape,
    num_classes: int,
    class_labels=None,
    ddim_steps:  int   = 50,
    w_eval:      float = 4.0,
    use_tag:     bool  = True,
    w_min:       float = 1.0,
    tag_beta:    float = 5.0,
    device:      str   = "cpu",
) -> torch.Tensor:
    """
    Generate samples via deterministic DDIM with classifier-free guidance.

    The guidance weight w(t) is either fixed (use_tag=False) or
    follows the TAG adaptive sigmoid schedule (use_tag=True).

    Args:
        model        UNet with forward(x, t, class_labels, force_class).
        schedule     DiffusionSchedule instance.
        shape        (B, C, H, W) shape of the output tensor.
        num_classes  Number of conditional classes; null token = num_classes.
        class_labels (B,) long tensor; sampled uniformly if None.
        ddim_steps   Number of DDIM denoising steps (default 50).
        w_eval       Peak guidance weight (w_max for TAG, fixed w otherwise).
        use_tag      If True, apply TAG adaptive guidance schedule (Eq. 3).
                     If False, use fixed w=w_eval at all timesteps.
        w_min        Minimum guidance weight for TAG (default 1.0).
        tag_beta     Sharpness of TAG sigmoid transition (default 5.0).
        device       Torch device string.

    Returns:
        (B, C, H, W) generated images in [-1, 1].
    """
    B = shape[0]
    T = schedule.timesteps

    if class_labels is None:
        class_labels = torch.randint(0, num_classes, (B,), device=device)

    null_labels = torch.full((B,), num_classes, device=device, dtype=torch.long)

    # Build DDIM timestep subsequence [0, step_ratio, 2*step_ratio, ...]
    step_ratio = T // ddim_steps
    ts         = torch.tensor(list(range(0, T, step_ratio)),
                              device=device, dtype=torch.long)
    ts_prev    = torch.cat([torch.tensor([-1], device=device), ts[:-1]])

    x = torch.randn(shape, device=device)

    for t_val, t_prev_val in zip(
        reversed(ts.tolist()),
        reversed(ts_prev.tolist())
    ):
        t_batch      = torch.full((B,), t_val,      device=device, dtype=torch.long)
        t_prev_batch = (torch.full((B,), t_prev_val, device=device, dtype=torch.long)
                        if t_prev_val >= 0 else None)

        # Two forward passes for CFG
        eps_cond   = model(x, t_batch, class_labels)
        eps_uncond = model(x, t_batch, null_labels)

        # Guidance weight: TAG adaptive or fixed
        if use_tag:
            w_t = tag_guidance_weight(t_val, T=T, w_min=w_min,
                                      w_max=w_eval, beta=tag_beta)
        else:
            w_t = w_eval

        # Composite guided prediction
        eps = eps_uncond + w_t * (eps_cond - eps_uncond)

        # DDIM update step (eta=0, fully deterministic)
        alpha_bar_t = schedule.extract(schedule.alphas_cumprod,
                                       t_batch, x.shape)
        if t_prev_batch is not None:
            alpha_bar_prev = schedule.extract(schedule.alphas_cumprod,
                                              t_prev_batch, x.shape)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        # Predicted x_0
        pred_x0 = torch.clamp(
            (x - torch.sqrt(1.0 - alpha_bar_t) * eps) /
            torch.sqrt(alpha_bar_t + 1e-8),
            -1.0, 1.0
        )

        # Direction pointing to x_t
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0)) * eps

        if t_prev_batch is not None:
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
        else:
            x = pred_x0

    return x

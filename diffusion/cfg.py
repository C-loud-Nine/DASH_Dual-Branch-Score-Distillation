# =============================================================================
# diffusion/cfg.py
# Timestep-Adaptive Guidance (TAG) schedule (Eq. 3 in paper).
#
# TAG modulates guidance strength at inference time only:
#   w(t) = w_min + (w_max - w_min) * sigmoid(beta * (0.5 - t/T))
#
# At t ≈ T (pure noise): w(t) ≈ w_min  — gentle guidance for structure.
# At t ≈ 0 (near-clean): w(t) ≈ w_max  — strong guidance for detail.
#
# TAG is applied at teacher inference only. Student distillation targets
# the branch outputs eps_T^y and eps_T^∅ directly, which are independent
# of w(t) (they come from individual forced forward passes, not from the
# composite guided prediction).
# =============================================================================

import torch


def tag_guidance_weight(
    t_val: int,
    T:     int   = 1000,
    w_min: float = 1.0,
    w_max: float = 4.0,
    beta:  float = 5.0,
) -> float:
    """
    Compute the scalar TAG guidance weight w(t) for a single timestep.

    Args:
        t_val  Scalar timestep (0 = clean, T = pure noise).
        T      Total diffusion steps (1000).
        w_min  Minimum guidance weight (applied at high noise, t ≈ T).
        w_max  Maximum guidance weight (applied at low noise,  t ≈ 0).
        beta   Sharpness of the sigmoid transition (5.0 in paper).

    Returns:
        Scalar float guidance weight.
    """
    alpha = torch.sigmoid(
        torch.tensor(beta * (0.5 - t_val / T), dtype=torch.float32)
    )
    return (w_min + (w_max - w_min) * alpha).item()


def tag_guidance_curve(
    T:     int   = 1000,
    w_min: float = 1.0,
    w_max: float = 4.0,
    beta:  float = 5.0,
) -> list:
    """
    Return TAG guidance weights for all timesteps 0..T-1.
    Useful for visualisation.

    Returns:
        List of T floats.
    """
    return [tag_guidance_weight(t, T=T, w_min=w_min, w_max=w_max, beta=beta)
            for t in range(T)]

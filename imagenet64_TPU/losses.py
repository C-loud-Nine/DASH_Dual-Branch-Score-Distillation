# =============================================================================
# losses.py — DASH dual-branch objective (main paper Eqs. 5-8).
#
#   L_im  = E[ omega_t || eps^y_S  - eps^y_T  ||^2 ]   conditional branch
#   L_un  = E[ omega_t || eps^0_S  - eps^0_T  ||^2 ]   unconditional branch
#   L_an  = E[ omega_t || eps^y_S  - eps      ||^2 ]   ground-truth anchor
#   L_DASH = lambda_im L_im + lambda_un L_un + lambda_an L_an
#
# L_im and L_un together specify both branch outputs on every sample, which
# reduces the zero-loss set of the composite objective from a d-dimensional
# subspace to a single point (main paper Sec. 3.4).
# =============================================================================

import torch


def _per_sample_sq(a, b):
    """||a - b||^2 per sample, shape (B,)."""
    return (a - b).pow(2).flatten(1).sum(1)


def dash_losses(eps_y_s, eps_u_s, eps_y_t, eps_u_t, eps_true, omega_hat):
    """Return (total, {components}).

    omega_hat: (B,) TIRT weights for this batch's timesteps, already
    normalised to unit mean over all T by the TIRT module.
    """
    l_im = (omega_hat * _per_sample_sq(eps_y_s, eps_y_t)).mean()
    l_un = (omega_hat * _per_sample_sq(eps_u_s, eps_u_t)).mean()
    l_an = (omega_hat * _per_sample_sq(eps_y_s, eps_true)).mean()
    return l_im, l_un, l_an


def composite_loss(eps_y_s, eps_u_s, eps_y_t, eps_u_t, omega_hat, w=4.0):
    """Baseline: supervise only the composite guided score (Meng et al.).

    Non-identifiable at the branch level -- any (eps^y_S, eps^0_S) on the
    d-dimensional subspace w d^y + (1-w) d^0 = 0 attains zero loss.
    """
    comp_s = eps_u_s + w * (eps_y_s - eps_u_s)
    comp_t = eps_u_t + w * (eps_y_t - eps_u_t)
    return (omega_hat * _per_sample_sq(comp_s, comp_t)).mean()


@torch.no_grad()
def calibration_metrics(eps_y_s, eps_u_s, eps_y_t, eps_u_t, eps_norm=1e-8):
    """Single-batch rho, cos(Delta), MSE_Delta (main paper Sec. 4.1); the
    reported values accumulate over all pairs (evaluate.py, calibration).

    Delta = eps^y - eps^0 is the guidance gap, the quantity CFG amplifies at
    every denoising step.
    """
    d_s = (eps_y_s - eps_u_s).flatten(1)
    d_t = (eps_y_t - eps_u_t).flatten(1)
    n_s, n_t = d_s.norm(dim=1), d_t.norm(dim=1)

    rho = (n_s.mean() / n_t.mean().clamp(min=eps_norm)).item()
    cos = ((d_s * d_t).sum(1) / (n_s * n_t).clamp(min=eps_norm)).mean().item()
    mse = (d_s - d_t).pow(2).mean(1).mean().item()
    return {"rho": rho, "cos_delta": cos, "mse_delta": mse}

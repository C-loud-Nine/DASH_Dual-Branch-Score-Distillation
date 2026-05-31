# =============================================================================
# training/losses.py
# DASH distillation objective (Eq. 4-7 in paper).
#
# Three losses, all weighted by the frozen TIRT curriculum omega_hat_t:
#
#   L_im  = E[omega_hat_t * || eps_S^y      - eps_T^y      ||^2]
#            Imitation loss: conditional branch matching.
#
#   L_un  = E[omega_hat_t * || eps_S^∅      - eps_T^∅      ||^2]
#            Unconditional loss: resolves structural underdetermination.
#            Dominant contribution (67%/62% of total gain on C10/C100).
#
#   L_an  = E[omega_hat_t * || eps_S^y      - eps            ||^2]
#            Anchor loss: regularises conditional branch toward ground-truth
#            noise, preventing drift from teacher error in early training.
#            Applied to conditional branch only — unconditional branch
#            already has an exact teacher target from L_un.
#
#   L_DASH = lam_im * L_im + lam_un * L_un + lam_an * L_an
#            Default weights: lam_im = lam_un = 1.0,  lam_an = 0.1
#
# TIRT weights are FROZEN (transferred from teacher checkpoint) and are
# used only for weighting, not updated during student training.
# =============================================================================

import torch
import torch.nn.functional as F


def tirt_weighted_mse(
    pred:           torch.Tensor,
    target:         torch.Tensor,
    t:              torch.Tensor,
    tirt_module,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """
    TIRT-weighted MSE between pred and target.

    Args:
        pred            (B, C, H, W)  student prediction.
        target          (B, C, H, W)  teacher or ground-truth target.
        t               (B,)          long tensor of timestep indices.
        tirt_module     Frozen TIRT instance (student_tirt).
        alphas_cumprod  (T,)          schedule tensor.

    Returns:
        Scalar weighted loss.
    """
    # Per-sample scalar MSE: mean over C*H*W
    loss_per = F.mse_loss(pred, target, reduction="none").view(
        t.shape[0], -1).mean(dim=1)
    return tirt_module(loss_per, t, alphas_cumprod)


def dash_loss(
    eps_S_cond:     torch.Tensor,
    eps_S_uncond:   torch.Tensor,
    eps_T_cond:     torch.Tensor,
    eps_T_uncond:   torch.Tensor,
    eps_true:       torch.Tensor,
    t:              torch.Tensor,
    tirt_module,
    alphas_cumprod: torch.Tensor,
    lam_im:         float = 1.0,
    lam_un:         float = 1.0,
    lam_an:         float = 0.1,
):
    """
    Full DASH distillation objective.

    Args:
        eps_S_cond      (B, C, H, W)  student conditional prediction eps_S^y.
        eps_S_uncond    (B, C, H, W)  student unconditional prediction eps_S^∅.
        eps_T_cond      (B, C, H, W)  frozen teacher conditional eps_T^y.
        eps_T_uncond    (B, C, H, W)  frozen teacher unconditional eps_T^∅.
        eps_true        (B, C, H, W)  ground-truth noise used to form x_t.
        t               (B,)          long tensor of timestep indices.
        tirt_module     Frozen TIRT instance.
        alphas_cumprod  (T,)          diffusion schedule tensor.
        lam_im          Weight for L_im (default 1.0).
        lam_un          Weight for L_un (default 1.0).
        lam_an          Weight for L_an (default 0.1).

    Returns:
        Tuple (L_total, L_im, L_un, L_an) of scalar tensors.
    """
    L_im = tirt_weighted_mse(eps_S_cond,   eps_T_cond,   t, tirt_module, alphas_cumprod)
    L_un = tirt_weighted_mse(eps_S_uncond, eps_T_uncond, t, tirt_module, alphas_cumprod)
    L_an = tirt_weighted_mse(eps_S_cond,   eps_true,     t, tirt_module, alphas_cumprod)

    L_total = lam_im * L_im + lam_un * L_un + lam_an * L_an
    return L_total, L_im, L_un, L_an

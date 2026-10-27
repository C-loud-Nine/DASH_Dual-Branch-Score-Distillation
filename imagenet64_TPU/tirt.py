# =============================================================================
# training/tirt.py
# Timestep-Importance Rebalanced Training (TIRT) and EMA helper.
# =============================================================================

from typing import Optional

import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    """Identity forward, negated gradient backward."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -grad_out


class TIRT(nn.Module):
    """Learnable per-timestep importance weights (paper Eq. 2).

        omega_t     = [ min(SNR_t, gamma) / SNR_t ] * sigmoid(lambda_t)
        omega_hat_t = omega_t / mean(omega)          # unit mean over all T

    The base factor is the eps-prediction form of Min-SNR. Min-SNR targets a
    weight of min(SNR, gamma) in x0-space; since L_x0 = SNR * L_eps, realising
    that target on an eps-prediction loss requires the division by SNR.

    lambda_t is initialised to 0, so sigmoid(0) = 0.5 uniformly and the
    schedule starts at the Min-SNR baseline up to a constant factor.
    The sigmoid bounds each raw factor to (0, 1); after unit-mean
    normalisation the relative weights are not bounded, so TIRT is a
    finite-budget heuristic (paper Sec. 3.3).

    Args:
        timesteps:        Total diffusion steps T.
        snr_gamma:        SNR cap gamma.
        weight_objective: "descent" minimises the weighted loss w.r.t.
                          lambda; "ascent" maximises it (gradient reversal,
                          so one optimiser suffices); "frozen" holds lambda
                          fixed, as used by TIRT Transfer in the student.
        alphas_cumprod:   Optional (T,) schedule. Supplying it precomputes the
                          base factor so the unit-mean normaliser is exact.
    """

    VALID_OBJECTIVES = ("descent", "ascent", "frozen")

    def __init__(
        self,
        timesteps: int = 1000,
        snr_gamma: float = 5.0,
        weight_objective: str = "descent",
        alphas_cumprod: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if weight_objective not in self.VALID_OBJECTIVES:
            raise ValueError(
                f"weight_objective must be one of {self.VALID_OBJECTIVES}, "
                f"got {weight_objective!r}"
            )

        self.importance_adjustment = nn.Parameter(torch.zeros(timesteps))
        self.timesteps = timesteps
        self.weight_objective = weight_objective

        if weight_objective == "frozen":
            self.importance_adjustment.requires_grad_(False)

        self.register_buffer("snr_gamma", torch.tensor(float(snr_gamma)))

        if alphas_cumprod is not None:
            self.register_buffer("base_all", self._base_from(alphas_cumprod))
        else:
            self.base_all = None

    def _base_from(self, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """min(SNR, gamma) / SNR over all T."""
        ab = alphas_cumprod.to(self.snr_gamma.device).clamp(1e-8, 1.0 - 1e-4)
        snr = ab / (1.0 - ab)
        return torch.minimum(snr, self.snr_gamma) / snr

    def _base(self, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        if self.base_all is not None:
            return self.base_all
        return self._base_from(alphas_cumprod)

    def _lam(self) -> torch.Tensor:
        lam = self.importance_adjustment
        if self.weight_objective == "ascent":
            lam = _GradReverse.apply(lam)
        return lam

    def full_schedule(self, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """Normalised omega_hat over all T timesteps, shape (T,)."""
        w = self._base(alphas_cumprod) * torch.sigmoid(self._lam())
        return w / w.mean().clamp(min=1e-12)

    def get_weights(
        self,
        t_batch: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """omega_hat for a batch of timesteps, shape (B,).

        Normalisation is over all T, so a timestep's weight does not depend
        on which other timesteps were sampled.
        """
        dev = self.snr_gamma.device
        t_batch = t_batch.to(dev)
        return self.full_schedule(alphas_cumprod.to(dev))[t_batch]

    def forward(
        self,
        loss_per_sample: torch.Tensor,
        t_batch: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """TIRT-weighted scalar loss."""
        w = self.get_weights(t_batch, alphas_cumprod).to(loss_per_sample.device)
        return (loss_per_sample * w).mean()

    @torch.no_grad()
    def export_lambda(self) -> torch.Tensor:
        """Converged lambda_t, for TIRT Transfer into the student."""
        return self.importance_adjustment.detach().clone()

    @torch.no_grad()
    def load_transferred(self, lam: torch.Tensor, freeze: bool = True):
        """Load a teacher lambda_t and, by default, freeze it.

        Under L_DASH the gradient on lambda measures student-teacher mismatch
        at a trained teacher rather than content difficulty, so the
        transferred schedule is held fixed during distillation.
        """
        if lam.shape != self.importance_adjustment.shape:
            raise ValueError(
                f"lambda shape {tuple(lam.shape)} != "
                f"{tuple(self.importance_adjustment.shape)}"
            )
        self.importance_adjustment.copy_(lam.to(self.importance_adjustment))
        if freeze:
            self.importance_adjustment.requires_grad_(False)
            self.weight_objective = "frozen"

    def extra_repr(self) -> str:
        return (
            f"timesteps={self.timesteps}, gamma={float(self.snr_gamma):.3g}, "
            f"weight_objective={self.weight_objective}"
        )


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMAHelper:
    """Exponential Moving Average of model parameters.

    Args:
        mu: EMA decay (0.9999 teacher, 0.9995 student).
    """

    def __init__(self, mu: float = 0.9999):
        self.mu = mu
        self.shadow = {}
        self.registered = False

    def register(self, module: nn.Module, force: bool = False):
        if self.registered and not force:
            return
        self.shadow = {
            n: p.data.clone().detach()
            for n, p in module.named_parameters()
            if p.requires_grad
        }
        self.registered = True

    def update(self, module: nn.Module):
        if not self.registered:
            self.register(module)
        with torch.no_grad():
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if n in self.shadow:
                    self.shadow[n].mul_(self.mu).add_(p.data, alpha=1.0 - self.mu)
                else:
                    self.shadow[n] = p.data.clone().detach()

    def ema_copy(self, module: nn.Module) -> nn.Module:
        copy_ = type(module)(module.config).to(next(module.parameters()).device)
        copy_.load_state_dict(module.state_dict())
        with torch.no_grad():
            for n, p in copy_.named_parameters():
                if p.requires_grad and n in self.shadow:
                    sd = self.shadow[n]
                    if p.device != sd.device:
                        sd = sd.to(p.device)
                    p.data.copy_(sd)
        return copy_

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "mu": self.mu, "registered": self.registered}

    def load_state_dict(self, sd: dict):
        self.shadow = {k: v.detach() for k, v in sd["shadow"].items()}
        self.mu = sd.get("mu", 0.9999)
        self.registered = sd.get("registered", True)

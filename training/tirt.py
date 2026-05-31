# =============================================================================
# training/tirt.py
# Timestep-Importance Rebalanced Training (TIRT) and EMA helper.
#
# TIRT (Eq. 2 in paper):
#   omega_t = min(SNR_t, gamma) * sigmoid(lambda_t)
#   omega_hat_t = omega_t / mean(omega)     (unit-mean normalisation)
#
# During teacher training, lambda_t is learned jointly with network
# parameters.  At convergence, omega_hat_t encodes data-adaptive
# difficulty — which timesteps remain hard after schedule extremes saturate.
#
# TIRT Transfer: the teacher's converged lambda_t is copied into the student
# and FROZEN throughout distillation.  Distillation gradients reward
# prediction-matching ease rather than content difficulty, so allowing
# lambda_t to retrain would corrupt the teacher curriculum.
# =============================================================================

import torch
import torch.nn as nn


class TIRT(nn.Module):
    """
    Learnable per-timestep importance weights for DDPM training.

    The weight formula follows Eq. 2:
        omega_t     = min(SNR_t, gamma) * sigmoid(lambda_t)
        omega_hat_t = omega_t / mean(omega)

    SNR_t = alpha_bar_t / (1 - alpha_bar_t) is computed from the
    diffusion schedule at call time; it is NOT stored here so this
    module can be used independently of the schedule object.

    Args:
        timesteps  Total diffusion steps T (1000).
        snr_gamma  SNR cap gamma (5.0 in paper).
    """

    def __init__(self, timesteps: int = 1000, snr_gamma: float = 5.0):
        super().__init__()
        # Learnable adjustment: initialised to 0 so sigmoid(0)=0.5
        # recovers the Min-SNR schedule at startup (warm start).
        self.importance_adjustment = nn.Parameter(torch.zeros(timesteps))
        self.timesteps             = timesteps
        self.register_buffer("snr_gamma", torch.tensor(snr_gamma))

    def get_weights(
        self,
        t_batch:        torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute normalised importance weights omega_hat for a batch of
        timesteps.

        Args:
            t_batch         (B,) long tensor of timestep indices.
            alphas_cumprod  (T,) float tensor from the diffusion schedule.

        Returns:
            (B,) normalised importance weights.
        """
        dev = self.snr_gamma.device
        if alphas_cumprod.device != dev:
            alphas_cumprod = alphas_cumprod.to(dev)
        if t_batch.device != dev:
            t_batch = t_batch.to(dev)

        alpha_bar = alphas_cumprod[t_batch]
        snr_t     = alpha_bar / (1.0 - alpha_bar + 1e-8)
        base_w    = torch.minimum(snr_t, self.snr_gamma)
        adj       = torch.sigmoid(self.importance_adjustment[t_batch])
        w         = base_w * adj
        return w / w.mean().clamp(min=1e-8)

    def forward(
        self,
        loss_per_sample: torch.Tensor,
        t_batch:         torch.Tensor,
        alphas_cumprod:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply TIRT weighting to per-sample losses and return the scalar mean.

        Args:
            loss_per_sample  (B,) per-sample MSE values.
            t_batch          (B,) long tensor of timestep indices.
            alphas_cumprod   (T,) float tensor from the diffusion schedule.

        Returns:
            Scalar weighted loss.
        """
        w = self.get_weights(t_batch, alphas_cumprod)
        return (loss_per_sample * w).mean()


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMAHelper:
    """
    Exponential Moving Average of model parameters.

    The shadow copy is updated after every training step.
    At evaluation time, ema_copy() instantiates a new model loaded with
    the shadow weights without modifying the training model.

    Args:
        mu  EMA decay rate (0.9999 for teacher, 0.9995 for student).
    """

    def __init__(self, mu: float = 0.9999):
        self.mu         = mu
        self.shadow     = {}
        self.registered = False

    def register(self, module: nn.Module, force: bool = False):
        """Initialise shadow with current parameter values."""
        if self.registered and not force:
            return
        self.shadow = {
            n: p.data.clone().detach()
            for n, p in module.named_parameters()
            if p.requires_grad
        }
        self.registered = True

    def update(self, module: nn.Module):
        """Update shadow: shadow = mu * shadow + (1-mu) * param."""
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
        """
        Return a new model instance with shadow (EMA) weights loaded.
        The training model is not modified.
        """
        copy_ = type(module)(module.config).to(
            next(module.parameters()).device)
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
        return {
            "shadow":     self.shadow,
            "mu":         self.mu,
            "registered": self.registered,
        }

    def load_state_dict(self, sd: dict):
        self.shadow     = {k: v.detach() for k, v in sd["shadow"].items()}
        self.mu         = sd.get("mu", 0.9999)
        self.registered = sd.get("registered", True)

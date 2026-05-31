# =============================================================================
# diffusion/schedule.py
# Cosine noise schedule and forward diffusion process (Ho et al. 2020).
#
# Used identically by teacher training and student distillation.
# All schedule tensors are registered on the calling device and accessed
# via the `extract` helper to align with batch timestep indices.
# =============================================================================

import math
import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal 2021, Eq. 17).

    Args:
        timesteps  Total number of diffusion steps T (1000).
        s          Offset for small-t smoothing.

    Returns:
        (T,) beta tensor clipped to [1e-4, 0.9999].
    """
    steps = timesteps + 1
    x     = torch.linspace(0, timesteps, steps)
    ac    = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac    = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 1e-4, 0.9999)
    return betas


class DiffusionSchedule:
    """
    Precomputed cosine schedule tensors, mirroring the exact schedule used
    in teacher training.  All tensors live on `device`.

    Attributes:
        betas                        (T,)
        alphas                       (T,)
        alphas_cumprod               (T,)   alpha_bar_t
        alphas_cumprod_prev          (T,)   alpha_bar_{t-1}
        sqrt_alphas_cumprod          (T,)
        sqrt_one_minus_alphas_cumprod (T,)
    """

    def __init__(self, timesteps: int = 1000, device: str = "cpu"):
        self.timesteps = timesteps
        self.device    = device

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas            = 1.0 - betas
        alphas_cumprod    = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas                         = betas
        self.alphas                        = alphas
        self.alphas_cumprod                = alphas_cumprod
        self.alphas_cumprod_prev           = alphas_cumprod_prev
        self.sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def extract(self, a: torch.Tensor, t: torch.Tensor,
                x_shape: tuple) -> torch.Tensor:
        """
        Gather schedule values at timestep indices t and reshape to
        broadcast over spatial dimensions.

        Args:
            a      (T,) schedule tensor.
            t      (B,) long tensor of timestep indices.
            x_shape tuple of (B, C, H, W).

        Returns:
            (B, 1, 1, 1) tensor for broadcasting.
        """
        if a.device != t.device:
            a = a.to(t.device)
        return a.gather(-1, t).reshape(
            t.shape[0], *((1,) * (len(x_shape) - 1))
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion: sample x_t given x_0 and timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0    (B, C, H, W)  clean image in [-1, 1].
            t     (B,)          long tensor of timesteps.
            noise (B, C, H, W)  pre-sampled noise; sampled if None.

        Returns:
            (B, C, H, W) noisy image x_t.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1ab = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1ab * noise

    def snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Signal-to-noise ratio at timestep t: SNR_t = alpha_bar_t / (1 - alpha_bar_t).
        Used by TIRT for per-timestep importance weighting.

        Args:
            t  (B,) or scalar long tensor.

        Returns:
            SNR values, same shape as t.
        """
        ab = self.alphas_cumprod[t.to(self.alphas_cumprod.device)]
        return ab / (1.0 - ab + 1e-8)

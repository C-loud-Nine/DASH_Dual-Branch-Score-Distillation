# =============================================================================
# models/unet.py
# ADM-style UNet with classifier-free guidance (CFG) conditioning.
# Shared by teacher (ch=128, nrb=2, ~35.8M params) and
#           student  (ch=64,  nrb=1, ~6.1M  params).
#
# Key design choices:
#   - GroupNorm(num_groups=32) throughout — required for weight compatibility
#     between teacher and student when transferring TIRT curriculum.
#   - Null class token at index num_classes enables unconditional forward pass
#     without a separate network.
#   - Self-attention applied only at 16x16 spatial resolution.
#   - force_class=True bypasses CFG dropout for distillation (branch targets).
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def nonlinearity(x):
    """Swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def Normalize(num_channels):
    """GroupNorm-32. num_groups=32 is hardcoded to keep teacher/student
    checkpoints weight-compatible across channel widths (64 and 128)."""
    return nn.GroupNorm(num_groups=32, num_channels=num_channels,
                        eps=1e-6, affine=True)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal timestep embedding (Vaswani et al.).

    Args:
        timesteps     (B,)  long tensor of diffusion timesteps.
        embedding_dim  int  output dimension.

    Returns:
        (B, embedding_dim) float tensor.
    """
    assert timesteps.ndim == 1
    half = embedding_dim // 2
    freq = math.log(10000) / (half - 1)
    freq = torch.exp(torch.arange(half, device=timesteps.device,
                                  dtype=torch.float32) * -freq)
    emb = timesteps.float()[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Upsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        if self.with_conv:
            return self.conv(F.pad(x, (0, 1, 0, 1), value=0))
        return F.avg_pool2d(x, 2, 2)


class ResnetBlock(nn.Module):
    """
    Pre-activation ResNet block with additive timestep+class conditioning.

    temb (timestep + class embedding) is projected and added after the
    first conv, following ADM (Dhariwal & Nichol 2021).
    """
    def __init__(self, *, in_channels, out_channels=None,
                 conv_shortcut=False, dropout=0.0, temb_channels=512):
        super().__init__()
        out = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out
        self.use_conv_shortcut = conv_shortcut

        self.norm1     = Normalize(in_channels)
        self.conv1     = nn.Conv2d(in_channels, out, 3, 1, 1)
        self.temb_proj = nn.Linear(temb_channels, out)
        self.norm2     = Normalize(out)
        self.dropout   = nn.Dropout(dropout)
        self.conv2     = nn.Conv2d(out, out, 3, 1, 1)

        if in_channels != out:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out, 1, 1, 0)

    def forward(self, x, temb):
        h = nonlinearity(self.norm1(x))
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.dropout(nonlinearity(self.norm2(h)))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = (self.conv_shortcut(x) if self.use_conv_shortcut
                 else self.nin_shortcut(x))
        return x + h


class AttnBlock(nn.Module):
    """Single-head self-attention (applied at 16x16 resolution only)."""
    def __init__(self, channels):
        super().__init__()
        self.norm     = Normalize(channels)
        self.q        = nn.Conv2d(channels, channels, 1)
        self.k        = nn.Conv2d(channels, channels, 1)
        self.v        = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        b, c, H, W = h.shape
        q = self.q(h).reshape(b, c, -1).permute(0, 2, 1)   # (B, HW, C)
        k = self.k(h).reshape(b, c, -1)                     # (B, C, HW)
        w = F.softmax(torch.bmm(q, k) * (c ** -0.5), dim=2) # (B, HW, HW)
        v = self.v(h).reshape(b, c, -1)                     # (B, C, HW)
        out = torch.bmm(v, w.permute(0, 2, 1)).reshape(b, c, H, W)
        return x + self.proj_out(out)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    ADM-style UNet for class-conditional diffusion (DDPM noise prediction).

    Teacher config : ch=128, ch_mult=(1,2,2,2), num_res_blocks=2  -> 35.8M
    Student config : ch=64,  ch_mult=(1,2,2,2), num_res_blocks=1  ->  6.1M

    Args (via config object):
        model.ch               Base channel width.
        model.ch_mult          Tuple of multipliers per resolution level.
        model.num_res_blocks   Residual blocks per encoder/decoder level.
        model.attn_resolutions Spatial resolutions where attention is applied.
        model.dropout          Dropout rate inside ResnetBlocks.
        model.in_channels      Input image channels (3 for RGB).
        model.out_ch           Output channels (3 for epsilon prediction).
        model.resamp_with_conv Use learned conv in up/downsample layers.
        image_size             Spatial resolution (32 for CIFAR).
        num_classes            Number of conditional classes.
        cfg_dropout            Probability of replacing label with null token
                               during training (classifier-free guidance).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        ch       = config.model.ch
        ch_mult  = config.model.ch_mult
        nrb      = config.model.num_res_blocks
        attn_r   = set(config.model.attn_resolutions)
        drop     = config.model.dropout
        resamp   = config.model.resamp_with_conv
        res      = config.image_size

        self.ch              = ch
        self.temb_ch         = ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks  = nrb

        # Timestep embedding: sin/cos -> Linear -> SiLU -> Linear
        self.temb       = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(ch, ch * 4),
            nn.Linear(ch * 4, ch * 4),
        ])

        # Class embedding. Index num_classes = unconditional null token.
        self.class_embed   = nn.Embedding(config.num_classes + 1, ch * 4)
        self.null_class_id = config.num_classes
        self.class_dropout = config.cfg_dropout

        # ── Encoder ──────────────────────────────────────────────────────────
        self.conv_in = nn.Conv2d(config.model.in_channels, ch, 3, 1, 1)
        in_ch_mult   = (1,) + tuple(ch_mult)
        b_in         = ch
        self.down    = nn.ModuleList()
        curr_res     = res

        for i in range(self.num_resolutions):
            blk  = nn.ModuleList()
            attn = nn.ModuleList()
            b_out = ch * ch_mult[i]

            for _ in range(nrb):
                blk.append(ResnetBlock(in_channels=b_in, out_channels=b_out,
                                       temb_channels=ch * 4, dropout=drop))
                b_in = b_out
                if curr_res in attn_r:
                    attn.append(AttnBlock(b_in))

            d       = nn.Module()
            d.block = blk
            d.attn  = attn
            if i != self.num_resolutions - 1:
                d.downsample = Downsample(b_in, resamp)
                curr_res //= 2
            self.down.append(d)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid         = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=b_in, out_channels=b_in,
                                       temb_channels=ch * 4, dropout=drop)
        self.mid.attn_1  = AttnBlock(b_in)
        self.mid.block_2 = ResnetBlock(in_channels=b_in, out_channels=b_in,
                                       temb_channels=ch * 4, dropout=drop)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            blk  = nn.ModuleList()
            attn = nn.ModuleList()
            b_out = ch * ch_mult[i]

            for j in range(nrb + 1):
                # +1 skip connection comes from encoder at matching resolution
                skip_in = ch * ch_mult[i] if j < nrb else ch * in_ch_mult[i]
                blk.append(ResnetBlock(in_channels=b_in + skip_in,
                                       out_channels=b_out,
                                       temb_channels=ch * 4, dropout=drop))
                b_in = b_out
                if curr_res in attn_r:
                    attn.append(AttnBlock(b_in))

            u       = nn.Module()
            u.block = blk
            u.attn  = attn
            if i != 0:
                u.upsample = Upsample(b_in, resamp)
                curr_res *= 2
            self.up.insert(0, u)

        self.norm_out = Normalize(b_in)
        self.conv_out = nn.Conv2d(b_in, config.model.out_ch, 3, 1, 1)

    def forward(self, x, t, class_labels=None, force_class=False):
        """
        Predict noise epsilon given noisy image x at timestep t.

        Args:
            x            (B, C, H, W)  noisy image x_t.
            t            (B,)          long tensor of timesteps.
            class_labels (B,)          long tensor; pass None for unconditional.
            force_class  bool          if True, disables CFG dropout (used
                                       during distillation to obtain the pure
                                       conditional/unconditional branch targets
                                       without stochastic dropout interference).

        Returns:
            (B, C, H, W) predicted noise tensor.
        """
        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = nonlinearity(self.temb.dense[0](temb))
        temb = self.temb.dense[1](temb)

        # Class conditioning
        if class_labels is None:
            # Unconditional pass: use null token
            class_labels = torch.full(
                (x.shape[0],), self.null_class_id,
                device=x.device, dtype=torch.long)
        elif self.training and self.class_dropout > 0 and not force_class:
            # CFG training dropout: randomly replace with null token
            if torch.rand(1).item() < self.class_dropout:
                class_labels = torch.full_like(class_labels, self.null_class_id)

        temb = temb + self.class_embed(class_labels)

        # Encoder
        hs = [self.conv_in(x)]
        for i in range(self.num_resolutions):
            for j in range(self.num_res_blocks):
                h = self.down[i].block[j](hs[-1], temb)
                if len(self.down[i].attn) > 0:
                    h = self.down[i].attn[j](h)
                hs.append(h)
            if i != self.num_resolutions - 1:
                hs.append(self.down[i].downsample(hs[-1]))

        # Bottleneck
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Decoder
        for i in reversed(range(self.num_resolutions)):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].block[j](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i].attn) > 0:
                    h = self.up[i].attn[j](h)
            if i != 0:
                h = self.up[i].upsample(h)

        return self.conv_out(nonlinearity(self.norm_out(h)))

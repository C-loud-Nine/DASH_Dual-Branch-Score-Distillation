# =============================================================================
# unet.py — ADM-style class-conditional UNet.
#
#   teacher: ch=128, num_res_blocks=3  ->  116.5M parameters
#   student: ch= 64, num_res_blocks=2  ->   22.3M parameters  (5.2x)
#
# Both use ch_mult (1,2,3,4) and self-attention at 16x16 and 8x8.
#
# force_class=True bypasses label dropout, which is required at distillation
# time: eps^y and eps^0 must come from separate deterministic passes so that
# L_im and L_un act as independent constraints on each branch.
# =============================================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], dim=-1)


def norm(c):
    return nn.GroupNorm(min(32, c), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.n1 = norm(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, cout)
        self.n2 = norm(cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = norm(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        att = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)
        h = (v @ att.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Block(nn.Module):
    """ResBlock, optionally followed by self-attention."""

    def __init__(self, cin, cout, emb_dim, dropout, attn):
        super().__init__()
        self.res = ResBlock(cin, cout, emb_dim, dropout)
        self.attn = AttnBlock(cout) if attn else None

    def forward(self, x, emb):
        x = self.res(x, emb)
        return self.attn(x) if self.attn is not None else x


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x, emb=None):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x, emb=None):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, cfg, num_classes, image_size):
        super().__init__()
        m = cfg
        ch, emb = m.ch, m.ch * 4
        self.ch = ch
        self.num_classes = num_classes
        nrb = m.num_res_blocks
        attn_res = set(m.attn_resolutions)
        nlev = len(m.ch_mult)

        self.t_emb = nn.Sequential(
            nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
        # +1 row is the null token used for the unconditional branch
        self.y_emb = nn.Embedding(num_classes + 1, emb)

        self.in_conv = nn.Conv2d(m.in_channels, ch, 3, padding=1)

        # ---- encoder: one flat list, skips pushed in the same order ----
        self.down = nn.ModuleList()
        skips = [ch]                      # channels of every tensor we push
        cur, res = ch, image_size
        for i, mult in enumerate(m.ch_mult):
            for _ in range(nrb):
                self.down.append(
                    Block(cur, ch * mult, emb, m.dropout, res in attn_res))
                cur = ch * mult
                skips.append(cur)
            if i < nlev - 1:
                self.down.append(Down(cur))
                skips.append(cur)
                res //= 2

        self.mid1 = ResBlock(cur, cur, emb, m.dropout)
        self.mid_attn = AttnBlock(cur)
        self.mid2 = ResBlock(cur, cur, emb, m.dropout)

        # ---- decoder: mirrors the encoder, consuming skips in reverse ----
        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(m.ch_mult))):
            for j in range(nrb + 1):
                self.up.append(
                    Block(cur + skips.pop(), ch * mult, emb,
                          m.dropout, res in attn_res))
                cur = ch * mult
            if i > 0:
                self.up.append(Up(cur))
                res *= 2

        self.out = nn.Sequential(
            norm(cur), nn.SiLU(), nn.Conv2d(cur, m.out_ch, 3, padding=1))

    def forward(self, x, t, y, force_class=False, cfg_dropout=0.0):
        """force_class=True disables label dropout (required at distillation)."""
        emb = self.t_emb(timestep_embedding(t, self.ch))
        if not force_class and cfg_dropout > 0.0 and self.training:
            drop = torch.rand(y.shape[0], device=y.device) < cfg_dropout
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        emb = emb + self.y_emb(y)

        h = self.in_conv(x)
        hs = [h]
        for mod in self.down:
            h = mod(h, emb)
            hs.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for mod in self.up:
            if isinstance(mod, Up):
                h = mod(h)
            else:
                h = mod(torch.cat([h, hs.pop()], dim=1), emb)
        return self.out(h)


def build(cfg_obj):
    """cfg_obj exposes .model, .num_classes, .image_size."""
    return UNet(cfg_obj.model, cfg_obj.num_classes, cfg_obj.image_size)

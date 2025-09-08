from __future__ import annotations

import torch


def make_offset_channel(x_masked: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Build per-shot normalized offset channel.
    x_masked : (B,1,H,W)
    offsets  : (B,H) float/long; will be cast to x_masked
    Returns  : (B,1,H,W)
    """
    B, _, H, W = x_masked.shape
    offs = offsets.to(x_masked)
    offs_c = offs - offs.median(dim=1, keepdim=True).values
    scale = offs_c.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    offs_n = offs_c / scale
    offs_ch = offs_n.unsqueeze(-1).expand(-1, -1, W).unsqueeze(1)
    return offs_ch

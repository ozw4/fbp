import torch

__all__ = ["make_offset_channel"]

def make_offset_channel(x_like: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Build a per-shot centered & scaled offset channel.

    Args:
        x_like: (B,1,H,W) tensor (provides dtype/device/shape references)
        offsets: (B,H) tensor of offsets per trace

    Returns:
        offs_ch: (B,1,H,W) tensor aligned with x_like

    """
    assert x_like.dim() == 4 and x_like.size(1) == 1, "x_like must be (B,1,H,W)"
    B, _, H, W = x_like.shape

    offs = offsets.to(device=x_like.device, dtype=x_like.dtype)
    assert offs.shape == (B, H), f"offsets must be (B,H), got {tuple(offs.shape)}"

    # center by per-shot median
    med = offs.median(dim=1, keepdim=True).values
    offs_c = offs - med

    # scale by per-shot max abs, floor=1.0 for safety
    denom = offs_c.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    offs_n = offs_c / denom  # (B,H) roughly in [-1,1]

    # expand to (B,1,H,W)
    offs_ch = offs_n.unsqueeze(-1).expand(-1, -1, W).unsqueeze(1).contiguous()
    return offs_ch

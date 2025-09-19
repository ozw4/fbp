import torch

__all__ = [
    "make_offset_channel",
    "make_offset_channel_phys",
    "make_time_channel",
]

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


def make_offset_channel_phys(
    x_like: torch.Tensor,
    offsets_m: torch.Tensor,
    x95_m: float,
    mode: str = "log1p",
    clip_hi: float = 1.5,
) -> torch.Tensor:
    """Build a physical offset channel with common normalization."""

    assert x_like.dim() == 4, "x_like must be (B,C,H,W)"
    B, _, H, W = x_like.shape

    offs = offsets_m.to(device=x_like.device, dtype=x_like.dtype)
    assert offs.shape == (B, H), f"offsets_m must be (B,H), got {tuple(offs.shape)}"

    offs = offs.clamp_min(0.0)
    assert x95_m > 0.0, "x95_m must be positive"
    scale = torch.as_tensor(x95_m, dtype=x_like.dtype, device=x_like.device)

    if mode == "log1p":
        denom = torch.log1p(scale)
        assert torch.all(denom > 0), "log1p(x95_m) must be positive"
        x = torch.log1p(offs) / denom
    elif mode == "linear":
        x = offs / scale
    else:  # pragma: no cover - guard against misconfiguration
        raise ValueError(f"Unsupported mode: {mode}")

    x = x.clamp_(min=0.0, max=clip_hi)
    x = x.unsqueeze(-1).expand(-1, -1, W).unsqueeze(1)
    return x.contiguous()


def make_time_channel(
    x_like: torch.Tensor,
    dt_sec: torch.Tensor | float,
    t95_ms: float,
    clip_hi: float = 1.5,
) -> torch.Tensor:
    """Build an absolute time channel normalized to a reference quantile."""

    assert x_like.dim() == 4, "x_like must be (B,C,H,W)"
    B, _, H, W = x_like.shape

    assert t95_ms > 0.0, "t95_ms must be positive"
    t95 = torch.as_tensor(t95_ms, dtype=x_like.dtype, device=x_like.device)

    dt_ms = torch.as_tensor(dt_sec, dtype=x_like.dtype, device=x_like.device) * 1000.0
    if dt_ms.ndim == 0:
        dt_ms = dt_ms.view(1).expand(B)
    else:
        assert dt_ms.shape[0] == B or dt_ms.numel() == 1, (
            "dt_sec must be scalar or have leading dim B",
        )
        if dt_ms.numel() == 1:
            dt_ms = dt_ms.view(1).expand(B)
    dt_ms = dt_ms.reshape(B, *dt_ms.shape[1:])
    while dt_ms.ndim < 4:
        dt_ms = dt_ms.unsqueeze(-1)

    time_idx = torch.arange(W, dtype=x_like.dtype, device=x_like.device).view(1, 1, 1, W)
    t_ms = dt_ms * time_idx
    t = (t_ms / t95).clamp_(min=0.0, max=clip_hi)
    t = t.expand(-1, -1, H, -1)
    return t.contiguous()

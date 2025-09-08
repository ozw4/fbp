# proc/util/velocity_mask.py
import math

import torch


@torch.no_grad()
def make_velocity_feasible_mask(
	offsets: torch.Tensor,  # (B,H) [m]
	dt_sec: torch.Tensor,  # (B,) or (B,1) [s]
	W: int,  # time bins (width)
	vmin: float,  # [m/s] slowest plausible (largest t)
	vmax: float,  # [m/s] fastest plausible (smallest t)
	t0_lo_ms: float = -20.0,  # early slack (can be negative)
	t0_hi_ms: float = 80.0,  # late slack
	taper_ms: float = 10.0,  # Hann taper half-width at both boundaries
	device=None,
	dtype=None,
) -> torch.Tensor:
	"""Return mask (B,H,W) in [0,1]: inside the velocity cone ~1, outside ~0,
	with optional Hann taper at the boundaries for smoothness.
	"""
	device = device or offsets.device
	dtype = dtype or offsets.dtype
	B, H = offsets.shape

	dt = dt_sec.to(device=device, dtype=dtype).view(B, 1, 1)  # (B,1,1)
	x = offsets.to(device=device, dtype=dtype).view(B, H, 1).abs()  # (B,H,1)

	t = torch.arange(W, device=device, dtype=dtype).view(1, 1, W) * dt  # (B,1,W)

	# Cone bounds in seconds (with slack)
	t_min = (x / max(vmax, 1e-6)) + (t0_lo_ms * 1e-3)
	t_max = (x / max(vmin, 1e-6)) + (t0_hi_ms * 1e-3)
	t_min = t_min.clamp_min(0.0)

	inside = (t >= t_min) & (t <= t_max)
	mask = inside.to(dtype)

	if taper_ms > 0:
		w = max(taper_ms * 1e-3, 1e-6)

		# lower transition: [t_min - w, t_min] : 0 -> 1
		lower = (t >= (t_min - w)) & (t < t_min)
		r_lo = ((t - (t_min - w)) / w).clamp(0.0, 1.0)
		hann_lo = 0.5 * (1.0 - torch.cos(math.pi * r_lo))
		mask = torch.where(lower, hann_lo.to(dtype), mask)

		# upper transition: [t_max, t_max + w] : 1 -> 0
		upper = (t > t_max) & (t <= (t_max + w))
		r_up = ((t_max + w - t) / w).clamp(0.0, 1.0)
		hann_up = 0.5 * (1.0 - torch.cos(math.pi * r_up))
		mask = torch.where(upper, hann_up.to(dtype), mask)

	return mask  # (B,H,W)


def apply_velocity_mask_to_logits(
	logits: torch.Tensor,  # (B,1,H,W)
	mask: torch.Tensor,  # (B,H,W)
	eps: float = 1e-12,
) -> torch.Tensor:
	"""Add log(mask) to logits so that outside the cone becomes ~ -inf (prob ~ 0)."""
	assert logits.dim() == 4 and logits.size(1) == 1
	logits[:, 0] = logits[:, 0] + torch.log(mask.clamp_min(eps))
	return logits

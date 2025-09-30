import math
from typing import Literal

import torch
from torch.amp.autocast_mode import autocast


from features import make_offset_channel


__all__ = ['cover_all_traces_predict', 'cover_all_traces_predict_chunked']


@torch.no_grad()
def cover_all_traces_predict(
	model: torch.nn.Module,
	x: torch.Tensor,
	*,
	mask_ratio: float = 0.5,
	noise_std: float = 1.0,
	mask_noise_mode: Literal['replace', 'add'] = 'replace',
	use_amp: bool = True,
	device=None,
	seed: int | None = 12345,
	passes_batch: int = 4,
	use_offset_input: bool = False,
	offsets: torch.Tensor | None = None,

) -> torch.Tensor:
	"""Predict each trace by covering all traces once.

	Args:
		mask_noise_mode: replace to overwrite, add to perturb traces.

	"""
	if x.dim() != 4 or x.size(1) < 1:
		raise AssertionError('x must be (B,>=1,H,W)')
	device = device or x.device

	B, C, H, W = x.shape

	m = max(1, min(int(round(mask_ratio * H)), H - 1))
	K = math.ceil(H / m)
	y_full = torch.empty_like(x[:, :1])
	static_channels = x[:, 1:] if C > 1 else None
	if use_offset_input:
		if offsets is not None:
			offs_ch = make_offset_channel(x[:, :1], offsets)
			static_channels = (
				offs_ch
				if static_channels is None or static_channels.size(1) == 0
				else torch.cat([static_channels, offs_ch], dim=1)
			)
		elif static_channels is None or static_channels.size(1) == 0:
			raise ValueError('offsets must be provided when use_offset_input=True')
	for b in range(B):
		if seed is not None:
			g = torch.Generator(device='cpu').manual_seed(seed + b)
			perm = torch.randperm(H, generator=g)
		else:
			perm = torch.randperm(H)
		chunks = [perm[i : i + m] for i in range(0, H, m)]
		for s in range(0, K, passes_batch):
			batch_chunks = chunks[s : s + passes_batch]
			xmb = []
			for idxs in batch_chunks:
				xm_data = x[b : b + 1, :1].clone()

				if seed is not None:
					gk = torch.Generator(device='cpu').manual_seed(
						(seed + b) * 100003 + s * 1009 + int(idxs[0])
					)
					n = (
						torch.randn((1, 1, len(idxs), W), generator=gk, device='cpu')
						* noise_std
					)
				else:
					n = torch.randn((1, 1, len(idxs), W), device='cpu') * noise_std
				n = n.to(device=device, non_blocking=True)
				idxs_dev = idxs.to(device)
				if mask_noise_mode == 'replace':
					xm_data[:, :, idxs_dev, :] = n
				elif mask_noise_mode == 'add':
					xm_data[:, :, idxs_dev, :] += n
				else:
					raise ValueError(f'Invalid mask_noise_mode: {mask_noise_mode}')

				if static_channels is not None and static_channels.size(1) > 0:
					static = static_channels[b : b + 1]
					xm = torch.cat([xm_data, static], dim=1)

				else:
					xm = xm_data
				xmb.append(xm)
			xmb = torch.cat(xmb, dim=0)
			dev_type = 'cuda' if xmb.is_cuda else 'cpu'
			with autocast(device_type=dev_type, enabled=use_amp):
				yb = model(xmb)
			for k, idxs in enumerate(batch_chunks):
				y_full[b, :, idxs.to(device), :] = yb[k, :, idxs.to(device), :]
	return y_full


@torch.no_grad()
def cover_all_traces_predict_chunked(
	model: torch.nn.Module,
	x: torch.Tensor,
	*,
	chunk_h: int = 128,
	overlap: int = 32,
	mask_ratio: float = 0.5,
	noise_std: float = 1.0,
	mask_noise_mode: Literal['replace', 'add'] = 'replace',
	use_amp: bool = True,
	device=None,
	seed: int = 12345,
	passes_batch: int = 4,

	use_offset_input: bool = False,
	offsets: torch.Tensor | None = None,

) -> torch.Tensor:
	"""Apply cover_all_traces_predict on tiled H-axis chunks.

	Args:
		mask_noise_mode: replace to overwrite, add to perturb traces.

	"""
	assert overlap < chunk_h, 'overlap は chunk_h より小さくしてください'
	device = device or x.device
	B, _, H, W = x.shape

	y_acc = torch.zeros_like(x[:, :1])

	w_acc = torch.zeros((B, 1, H, 1), dtype=x.dtype, device=device)
	step = chunk_h - overlap
	s = 0
	while s < H:
		e = min(s + chunk_h, H)
		xt = x[:, :, s:e, :]

		offs_t = offsets[:, s:e] if offsets is not None else None

		yt = cover_all_traces_predict(
			model,
			xt,
			mask_ratio=mask_ratio,
			noise_std=noise_std,
			mask_noise_mode=mask_noise_mode,
			use_amp=use_amp,
			device=device,
			seed=seed + s,
			passes_batch=passes_batch,
			use_offset_input=use_offset_input,
			offsets=offs_t,

		)
		h_t = e - s
		w = torch.ones((1, 1, h_t, 1), dtype=x.dtype, device=device)
		left_ov = min(overlap, s)
		right_ov = min(overlap, H - e)
		if left_ov > 0:
			ramp = torch.linspace(
				0, 1, steps=left_ov, device=device, dtype=x.dtype
			).view(1, 1, -1, 1)
			w[:, :, :left_ov, :] = ramp
		if right_ov > 0:
			ramp = torch.linspace(
				1, 0, steps=right_ov, device=device, dtype=x.dtype
			).view(1, 1, -1, 1)
			w[:, :, -right_ov:, :] = ramp
		y_acc[:, :, s:e, :] += yt * w
		w_acc[:, :, s:e, :] += w
		if e == H:
			break
		s += step
	y_full = y_acc / (w_acc + 1e-8)
	return y_full

import torch

__all__ = ['make_mask_2d', 'segy_collate']

def make_mask_2d(mask_indices_list, H, W, device):
	"""Create (B,1,H,W) mask with ones on masked traces."""
	B = len(mask_indices_list)
	mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
	for b, idxs in enumerate(mask_indices_list):
		if len(idxs) == 0:
			continue
		mask[b, 0, idxs, :] = 1.0
	return mask

def segy_collate(batch, task: str = 'recon', label_sigma: float = 5.0):
	"""Collate function for MaskedSegyGather outputs.

	Parameters
	----------
	batch:
	    Iterable of samples from :class:`MaskedSegyGather`.
	task:
	    ``'recon'`` for reconstruction (identity target) or ``'fb_seg'`` for
	    first-break segmentation where a Gaussian target is generated.
	label_sigma:
	    Standard deviation of the Gaussian in samples. A lower bound of
	    ``1e-6`` is enforced for numerical stability.

	"""
	x_masked = torch.stack([b['masked'] for b in batch], dim=0)
	x_orig = torch.stack([b['original'] for b in batch], dim=0)
	B, _, H, W = x_masked.shape

	# create mask of randomly masked traces
	mask_2d = torch.zeros((B, 1, H, W), dtype=x_masked.dtype)
	for i, b in enumerate(batch):
		idxs = b['mask_indices']
		if idxs:
			mask_2d[i, 0, idxs, :] = 1.0

	fb_idx = torch.stack([b['fb_idx'] for b in batch], dim=0)

	if task == 'fb_seg':
		sigma = max(float(label_sigma), 1e-6)
		t = torch.arange(W, dtype=x_masked.dtype).view(1, 1, 1, W)
		t0 = fb_idx.to(x_masked.dtype).view(B, 1, H, 1)
		gauss = torch.exp(-0.5 * ((t - t0) / sigma) ** 2)
		valid = (fb_idx >= 0).view(B, 1, H, 1)
		target = gauss * valid
	else:
		target = x_orig

	meta = {
		'file_path': [b['file_path'] for b in batch],
		'key_name': [b['key_name'] for b in batch],
		'indices': [b['indices'] for b in batch],
		'mask_indices': [b['mask_indices'] for b in batch],
		'fb_idx': fb_idx,
	}
	return x_masked, target, mask_2d, meta

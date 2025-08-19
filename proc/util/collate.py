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

def segy_collate(batch):
	"""Collate function for MaskedSegyGather outputs."""
	x_masked = torch.stack([b['masked'] for b in batch], dim=0)
	x_orig = torch.stack([b['original'] for b in batch], dim=0)
	B, _, H, W = x_masked.shape
	mask_2d = torch.zeros((B, 1, H, W), dtype=x_masked.dtype)
	for i, b in enumerate(batch):
		idxs = b['mask_indices']
		if idxs:
			mask_2d[i, 0, idxs, :] = 1.0
	fb_idx = torch.stack([b['fb_idx'] for b in batch], dim=0)
	meta = {
		'file_path': [b['file_path'] for b in batch],
		'key_name': [b['key_name'] for b in batch],
		'indices': [b['indices'] for b in batch],
		'mask_indices': [b['mask_indices'] for b in batch],
		'fb_idx': fb_idx,
	}
	return x_masked, x_orig, mask_2d, meta

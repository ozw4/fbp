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
        B, _, H, W = x_masked.shape
        if 'target' in batch[0]:
                teacher = torch.stack([b['target'] for b in batch], dim=0)
                mask = None
        else:
                teacher = torch.stack([b['original'] for b in batch], dim=0)
                mask_indices_list = [b.get('mask_indices', []) for b in batch]
                mask = make_mask_2d(mask_indices_list, H, W, device=x_masked.device)
        fb_idx = torch.stack([b['fb_idx'] for b in batch], dim=0)
        offsets = torch.stack([b['offsets'] for b in batch], dim=0)
        dt_sec = torch.stack([b['dt_sec'] for b in batch], dim=0)
        meta = {
                'file_path': [b['file_path'] for b in batch],
                'key_name': [b['key_name'] for b in batch],
                'indices': [b['indices'] for b in batch],
                'mask_indices': [b.get('mask_indices', []) for b in batch],
                'fb_idx': fb_idx,
                'offsets': offsets,
                'dt_sec': dt_sec,
        }
        return x_masked, teacher, mask, meta

from collections.abc import Sequence

import numpy as np
import segyio
import torch

__all__ = ['_read_gather_by_indices', 'load_synth_pair']

def _read_gather_by_indices(
	f: segyio.SegyFile,
	indices: np.ndarray,
	*,
	target_len: int,
) -> np.ndarray:
	"""Read traces and align to (H, target_len)."""
	traces = []
	for i in np.sort(indices):
		a = np.asarray(f.trace[i], dtype=np.float32)
		if a.shape[0] >= target_len:
			a = a[:target_len]
		else:
			pad = target_len - a.shape[0]
			a = np.pad(a, (0, pad), mode='constant')
		traces.append(a)
	if len(traces) == 0:
		return np.empty((0, target_len), dtype=np.float32)
	return np.stack(traces, axis=0)

def load_synth_pair(
	noisy_path: str,
	clean_path: str,
	*,
	extract_key1idxs: Sequence[int] | int = (401,),
	target_len: int = 6016,
	standardize: bool = True,
	endian: str = 'little',
):
	"""Load synthetic noisy/clean pair and return padded tensors."""
	if isinstance(extract_key1idxs, int):
		extract_key1idxs = [extract_key1idxs]
	used_ffids = []
	gathers_noisy = []
	gathers_clean = []
	H_list = []
	with (
		segyio.open(noisy_path, 'r', ignore_geometry=True, strict=False, endian=endian) as fn,
		segyio.open(clean_path, 'r', ignore_geometry=True, strict=False, endian=endian) as fc,
	):
		cdp_tr = np.asarray(fn.attributes(segyio.TraceField.CDP_TRACE)[:])
		for ff in extract_key1idxs:
			idx = np.where(cdp_tr == ff)[0]
			g_noisy = _read_gather_by_indices(fn, idx, target_len=target_len)
			g_clean = _read_gather_by_indices(fc, idx, target_len=target_len)
			if standardize and g_noisy.shape[0] > 0:
				for x in (g_noisy, g_clean):
					m = x.mean(axis=1, keepdims=True)
					s = x.std(axis=1, keepdims=True) + 1e-10
					x -= m
					x /= s
			gathers_noisy.append(g_noisy)
			gathers_clean.append(g_clean)
			used_ffids.append(int(ff))
			H_list.append(int(g_noisy.shape[0]))
	if len(gathers_noisy) == 0:
		raise ValueError('指定した FFID がいずれのファイルにも見つかりませんでした。')
	N = len(gathers_noisy)
	Hmax = max(H_list)
	W = target_len
	x_noisy = torch.zeros((N, 1, Hmax, W), dtype=torch.float32)
	x_clean = torch.zeros((N, 1, Hmax, W), dtype=torch.float32)
	valid_mask = torch.zeros((N, 1, Hmax, 1), dtype=torch.float32)
	for i, (gn, gc, H) in enumerate(zip(gathers_noisy, gathers_clean, H_list, strict=False)):
		x_noisy[i, 0, :H, :] = torch.from_numpy(gn)
		x_clean[i, 0, :H, :] = torch.from_numpy(gc)
		valid_mask[i, 0, :H, 0] = 1.0
	return x_noisy, x_clean, valid_mask, used_ffids, H_list

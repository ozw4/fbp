import random
import warnings
from fractions import Fraction
from typing import Literal

import numpy as np
import segyio
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .augment import (
	_apply_freq_augment,
	_spatial_stretch_sameH,
)

__all__ = ['MaskedSegyGather']


class MaskedSegyGather(Dataset):
	"""Dataset reading SEG-Y gathers with optional augmentation."""

	def __init__(
		self,
		segy_files: list[str],
		fb_files: list[str],
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		mask_ratio: float = 0.5,
		mask_mode: Literal['replace', 'add'] = 'replace',
		mask_noise_std: float = 1.0,
		pick_ratio: float = 0.3,
		target_len: int = 6016,
		flip: bool = False,
		augment_time_prob: float = 0.0,
		augment_time_range: tuple[float, float] = (0.95, 1.05),
		augment_space_prob: float = 0.0,
		augment_space_range: tuple[float, float] = (0.90, 1.10),
		augment_freq_prob: float = 0.0,
		augment_freq_kinds: tuple[str, ...] = ('bandpass', 'lowpass', 'highpass'),
		augment_freq_band: tuple[float, float] = (0.05, 0.45),
		augment_freq_width: tuple[float, float] = (0.10, 0.35),
		augment_freq_roll: float = 0.02,
		augment_freq_restandardize: bool = True,
		target_mode: Literal['recon', 'fb_seg'] = 'recon',
		label_sigma: float = 1.0,
	) -> None:
		"""Initialize dataset.

		Args:
			mask_mode: replace to overwrite, add to perturb traces.
			mask_noise_std: standard deviation of masking noise.

		"""
		self.segy_files = segy_files
		self.fb_files = fb_files
		self.ffid_byte = ffid_byte
		self.chno_byte = chno_byte
		self.mask_ratio = mask_ratio
		self.mask_mode = mask_mode
		self.mask_noise_std = mask_noise_std
		self.flip = flip
		self.pick_ratio = pick_ratio
		self.target_len = target_len
		self.augment_time_prob = augment_time_prob
		self.augment_time_range = augment_time_range
		self.augment_space_prob = augment_space_prob
		self.augment_space_range = augment_space_range
		self.augment_freq_prob = augment_freq_prob
		self.augment_freq_kinds = augment_freq_kinds
		self.augment_freq_band = augment_freq_band
		self.augment_freq_width = augment_freq_width
		self.augment_freq_roll = augment_freq_roll
		self.augment_freq_restandardize = augment_freq_restandardize
		self.target_mode = target_mode
		self.label_sigma = label_sigma
		self.file_infos = []
		for segy_path, fb_path in zip(self.segy_files, self.fb_files, strict=False):
			print(f'Loading {segy_path} and {fb_path}')
			f = segyio.open(segy_path, 'r', ignore_geometry=True)
			mmap = f.trace.raw[:]
			ffid_values = f.attributes(self.ffid_byte)[:]
			ffid_key_to_indices = self._build_index_map(ffid_values)
			ffid_unique_keys = list(ffid_key_to_indices.keys())
			chno_values = f.attributes(self.chno_byte)[:]
			chno_key_to_indices = self._build_index_map(chno_values)
			chno_unique_keys = list(chno_key_to_indices.keys())
			dt = int(f.bin[segyio.BinField.Interval]) / 1e3
			try:
				offsets = f.attributes(segyio.TraceField.offset)[:]
				offsets = np.asarray(offsets, dtype=np.float32)
				if len(offsets) != f.tracecount:
					warnings.warn(
						f'offset length mismatch in {segy_path}',
					)
					offsets = np.zeros(f.tracecount, dtype=np.float32)
			except Exception as e:
				warnings.warn(
					f'failed to read offsets from {segy_path}: {e}',
				)
				offsets = np.zeros(f.tracecount, dtype=np.float32)
			fb = np.load(fb_path)
			self.file_infos.append(
				dict(
					path=segy_path,
					mmap=mmap,
					ffid_values=ffid_values,
					ffid_key_to_indices=ffid_key_to_indices,
					ffid_unique_keys=ffid_unique_keys,
					chno_values=chno_values,
					chno_key_to_indices=chno_key_to_indices,
					chno_unique_keys=chno_unique_keys,
					n_samples=f.samples.size,
					n_traces=f.tracecount,
					dt=dt,
					segy_obj=f,
					fb=fb,
					offsets=offsets,
				)
			)

	def close(self) -> None:
		"""Close all opened SEG-Y file objects."""
		for info in self.file_infos:
			segy_obj = info.get('segy_obj')
			if segy_obj is not None:
				try:
					segy_obj.close()
				except Exception:
					pass
		self.file_infos.clear()

	def __del__(self) -> None:
		self.close()

	def _fit_time_len(
		self, x: np.ndarray, start: int | None = None
	) -> tuple[np.ndarray, int]:
		T, target = x.shape[1], self.target_len
		if start is None:
			start = np.random.randint(0, max(1, T - target + 1)) if target < T else 0
		if target < T:
			return x[:, start : start + target], start
		if target > T:
			pad = target - T
			return np.pad(x, ((0, 0), (0, pad)), mode='constant'), start
		return x, start

	def _build_index_map(self, key_array: np.ndarray) -> dict[int, np.ndarray]:
		uniq, inv, counts = np.unique(
			key_array, return_inverse=True, return_counts=True
		)
		sort_idx = np.argsort(inv, kind='mergesort')
		split_points = np.cumsum(counts)[:-1]
		groups = np.split(sort_idx, split_points)
		return {int(k): g.astype(np.int32) for k, g in zip(uniq, groups, strict=False)}

	def __len__(self) -> int:
		return 10**6

	def __getitem__(self, _=None):
		while True:
			info = random.choice(self.file_infos)
			mmap = info['mmap']
			fb = info['fb']
			key_name = random.choice(['ffid', 'chno'])
			unique_keys = info[f'{key_name}_unique_keys']
			key_to_indices = info[f'{key_name}_key_to_indices']
			key = random.choice(unique_keys)
			indices = key_to_indices[key]
			n_total = len(indices)
			if n_total >= 128:
				start_idx = random.randint(0, n_total - 128)
				selected_indices = indices[start_idx : start_idx + 128]
				pad_len = 0
			else:
				selected_indices = indices
				pad_len = 128 - n_total
			fb_subset = fb[selected_indices]
			if pad_len > 0:
				fb_subset = np.concatenate(
					[fb_subset, np.zeros(pad_len, dtype=fb_subset.dtype)]
				)
			offsets_full = info['offsets']
			off_subset = offsets_full[selected_indices].astype(np.float32, copy=False)
			if pad_len > 0:
				off_subset = np.concatenate(
					[off_subset, np.zeros(pad_len, dtype=np.float32)]
				)
			pick_ratio = np.count_nonzero(fb_subset > 0) / len(fb_subset)
			if pick_ratio >= self.pick_ratio:
				break
		x = mmap[selected_indices].astype(np.float32, copy=False)
		if pad_len > 0:
			pad_tr = np.zeros((pad_len, x.shape[1]), dtype=np.float32)
			x = np.concatenate([x, pad_tr], axis=0)
		x = x - np.mean(x, axis=1, keepdims=True)
		x = x / (np.std(x, axis=1, keepdims=True) + 1e-10)
		if self.flip and random.random() < 0.5:
			x = np.flip(x, axis=0).copy()
			fb_subset = fb_subset[::-1].copy()
			off_subset = off_subset[::-1].copy()
		factor = 1.0
		if self.augment_time_prob > 0 and random.random() < self.augment_time_prob:
			factor = random.uniform(*self.augment_time_range)
			frac = Fraction(factor).limit_denominator(128)
			up, down = frac.numerator, frac.denominator
			H_tmp = x.shape[0]
			x = np.stack(
				[resample_poly(x[h], up, down, padtype='line') for h in range(H_tmp)],
				axis=0,
			)
		x, start = self._fit_time_len(x)
		did_space = False
		f_h = 1.0
		if self.augment_space_prob > 0 and random.random() < self.augment_space_prob:
			f_h = random.uniform(*self.augment_space_range)
			x = _spatial_stretch_sameH(x, f_h)  # 既存行
			off_subset = _spatial_stretch_sameH(off_subset[:, None], f_h)[:, 0].astype(
				np.float32,
				copy=False,
			)
			did_space = True
		if self.augment_freq_prob > 0 and random.random() < self.augment_freq_prob:
			x = _apply_freq_augment(
				x,
				self.augment_freq_kinds,
				self.augment_freq_band,
				self.augment_freq_width,
				self.augment_freq_roll,
				self.augment_freq_restandardize,
			)
		fb_idx_win = np.floor(fb_subset * factor).astype(np.int64) - start
		invalid = (fb_idx_win <= 0) | (fb_idx_win >= self.target_len)
		fb_idx_win[invalid] = -1
		if self.target_mode == 'recon':
			H = x.shape[0]
			num_mask = int(self.mask_ratio * H)
			mask_idx = random.sample(range(H), num_mask) if num_mask > 0 else []
			x_masked = x.copy()
			if num_mask > 0:
				noise = np.random.normal(
					0.0, self.mask_noise_std, size=(num_mask, x.shape[1])
				)
				if self.mask_mode == 'replace':
					x_masked[mask_idx] = noise
				elif self.mask_mode == 'add':
					x_masked[mask_idx] += noise
				else:
					raise ValueError(f'Invalid mask_mode: {self.mask_mode}')
		else:
			mask_idx = []
			x_masked = x.copy()
		if self.target_mode == 'fb_seg':
			sigma = max(float(self.label_sigma), 1e-6)
			H_t, W_t = x.shape
			t = np.arange(W_t, dtype=np.float32)[None, :]
			target = np.zeros((H_t, W_t), dtype=np.float32)

			idx = fb_idx_win
			valid = idx >= 0
			if valid.any():
				idxv = idx[valid].astype(np.float32)[:, None]
				g = np.exp(-0.5 * ((t - idxv) / sigma) ** 2)
				g /= g.max(axis=1, keepdims=True) + 1e-12
				target[valid] = g

			# ② ターゲットにも同じ空間ストレッチを適用
			if did_space:
				target = _spatial_stretch_sameH(target, f_h)

			target_t = torch.from_numpy(target)[None, ...]
		x_t = torch.from_numpy(x)[None, ...]
		xm = torch.from_numpy(x_masked)[None, ...]
		fb_idx_t = torch.from_numpy(fb_idx_win)
		off_t = torch.from_numpy(off_subset)
		sample = {
			'masked': xm,
			'original': x_t,
			'fb_idx': fb_idx_t,
			'offsets': off_t,
			'key_name': key_name,
			'indices': selected_indices,
			'file_path': info['path'],
		}
		if self.target_mode == 'recon':
			sample['mask_indices'] = mask_idx
		else:
			sample['target'] = target_t
		return sample

import random
import warnings
from fractions import Fraction
from pathlib import Path
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


def _load_headers_with_cache(
	segy_path: str,
	ffid_byte,
	chno_byte,
	cmp_byte=None,
	cache_dir: str | None = None,
	rebuild: bool = False,
):
	segy_p = Path(segy_path)
	cache_p = (
		Path(cache_dir) / (segy_p.name + '.headers.npz')
		if cache_dir
		else segy_p.with_suffix(segy_p.suffix + '.headers.npz')
	)

	# 既存かつ新しければキャッシュを使う

	try:
		if (
			(not rebuild)
			and cache_p.exists()
			and cache_p.stat().st_mtime >= segy_p.stat().st_mtime
		):
			print(
				rebuild,
				cache_p.exists(),
				cache_p.stat().st_mtime >= segy_p.stat().st_mtime,
			)
			z = np.load(cache_p, allow_pickle=False)
			meta = {
				'ffid_values': z['ffid_values'],
				'chno_values': z['chno_values'],
				'cmp_values': (z['cmp_values'] if 'cmp_values' in z.files else None),
				'offsets': z['offsets'],
				'dt_us': int(z['dt_us']),
				'n_traces': int(z['n_traces']),
				'n_samples': int(z['n_samples']),
			}
			print(f'Loaded header cache from {cache_p}')
			return meta
	except Exception:
		# 壊れている等は作り直す
		pass

	# キャッシュ無 or 不正 → segyio で読み直し
	with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
		ffid_values = np.asarray(f.attributes(ffid_byte)[:], dtype=np.int32)
		chno_values = np.asarray(f.attributes(chno_byte)[:], dtype=np.int32)
		cmp_values = None
		if cmp_byte is not None:
			try:
				cmp_values = np.asarray(f.attributes(cmp_byte)[:], dtype=np.int32)
			except Exception:
				cmp_values = None

		try:
			offsets = np.asarray(
				f.attributes(segyio.TraceField.offset)[:], dtype=np.float32
			)
			if len(offsets) != f.tracecount:
				warnings.warn(f'offset length mismatch in {segy_path}')
				offsets = np.zeros(f.tracecount, dtype=np.float32)
		except Exception:
			warnings.warn(f'failed to read offsets from {segy_path}')
			offsets = np.zeros(f.tracecount, dtype=np.float32)

		dt_us = int(f.bin[segyio.BinField.Interval])
		meta = dict(
			ffid_values=ffid_values,
			chno_values=chno_values,
			cmp_values=(
				cmp_values if cmp_values is not None else np.array([], dtype=np.int32)
			),
			offsets=offsets,
			dt_us=dt_us,
			n_traces=f.tracecount,
			n_samples=f.samples.size,
		)

	# 保存（一時ファイル→置換で安全に）
	try:
		tmp = cache_p.with_name(cache_p.stem + '.tmp' + cache_p.suffix)
		np.savez_compressed(tmp, **meta)
		print(f'Saved header cache to {cache_p}')
		tmp.replace(cache_p)
	except Exception:
		pass

	# 返却整形
	meta['cmp_values'] = (
		None
		if (isinstance(meta['cmp_values'], np.ndarray) and meta['cmp_values'].size == 0)
		else meta['cmp_values']
	)
	return meta


class MaskedSegyGather(Dataset):
	"""Dataset reading SEG-Y gathers with optional augmentation."""

	def __init__(
		self,
		segy_files: list[str],
		fb_files: list[str],
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		cmp_byte=segyio.TraceField.CDP,
		primary_keys: tuple[str, ...]
		| None = None,  # 例: ('ffid','chno','cmp') / ('ffid',)
		primary_key_weights: tuple[float, ...] | None = None,
		use_superwindow: bool = False,
		sw_halfspan: int = 0,
		sw_prob: float = 0.3,
		use_header_cache: bool = False,
		header_cache_dir: str | None = None,
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
		self.cmp_byte = cmp_byte
		self.primary_keys = tuple(primary_keys) if primary_keys else None
		self.primary_key_weights = (
			tuple(primary_key_weights) if primary_key_weights else None
		)
		self.use_superwindow = use_superwindow
		self.sw_halfspan = int(sw_halfspan)
		self.sw_prob = sw_prob
		self.use_header_cache = use_header_cache
		self.header_cache_dir = header_cache_dir

		self._valid_primary_keys = {'ffid', 'chno', 'cmp'}
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
			print(self.use_header_cache, self.header_cache_dir)
			if self.use_header_cache:
				meta = _load_headers_with_cache(
					segy_path,
					self.ffid_byte,
					self.chno_byte,
					self.cmp_byte,
					cache_dir=self.header_cache_dir,
					rebuild=False,  # 必要なら True に
				)
				ffid_values = meta['ffid_values']
				chno_values = meta['chno_values']
				cmp_values = meta['cmp_values']
				offsets = meta['offsets']
				dt_us = meta['dt_us']
				n_traces = meta['n_traces']
				n_samples = meta['n_samples']
				dt = dt_us / 1e3
				dt_sec = dt_us * 1e-6
			else:
				# 従来の読み方（そのまま）
				f_tmp = segyio.open(segy_path, 'r', ignore_geometry=True)
				ffid_values = f_tmp.attributes(self.ffid_byte)[:]
				chno_values = f_tmp.attributes(self.chno_byte)[:]
				cmp_values = None
				if self.cmp_byte is not None:
					try:
						cmp_values = f_tmp.attributes(self.cmp_byte)[:]
					except Exception as e:
						warnings.warn(f'CMP header not available for {segy_path}: {e}')
						cmp_values = None
				dt_us = int(f_tmp.bin[segyio.BinField.Interval])
				dt = dt_us / 1e3
				dt_sec = dt_us * 1e-6
				try:
					offsets = f_tmp.attributes(segyio.TraceField.offset)[:]
					offsets = np.asarray(offsets, dtype=np.float32)
					if len(offsets) != f_tmp.tracecount:
						warnings.warn(f'offset length mismatch in {segy_path}')
						offsets = np.zeros(f_tmp.tracecount, dtype=np.float32)
				except Exception as e:
					warnings.warn(f'failed to read offsets from {segy_path}: {e}')
					offsets = np.zeros(f_tmp.tracecount, dtype=np.float32)
				n_traces = f_tmp.tracecount
				n_samples = f_tmp.samples.size
				f_tmp.close()
			# ▲▲ ここまでヘッダ取得 ▲▲

			# 以降は従来どおり：mmap用に開いて保持
			f = segyio.open(segy_path, 'r', ignore_geometry=True)
			mmap = f.trace.raw[:]

			ffid_key_to_indices = self._build_index_map(ffid_values)
			ffid_unique_keys = list(ffid_key_to_indices.keys())
			chno_key_to_indices = self._build_index_map(chno_values)
			chno_unique_keys = list(chno_key_to_indices.keys())
			cmp_key_to_indices = (
				self._build_index_map(cmp_values) if (cmp_values is not None) else None
			)
			cmp_unique_keys = (
				list(cmp_key_to_indices.keys())
				if (cmp_key_to_indices is not None)
				else None
			)

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
					cmp_values=cmp_values,
					cmp_key_to_indices=cmp_key_to_indices,
					cmp_unique_keys=cmp_unique_keys,
					n_samples=n_samples,
					n_traces=n_traces,
					dt=dt,
					dt_sec=dt_sec,
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
			cmp_available = (
				bool(info.get('cmp_unique_keys'))
				and isinstance(info['cmp_unique_keys'], (list, tuple))
				and len(info['cmp_unique_keys']) > 0
			)

			# 1) Hydra 指定がある場合は、それを優先して候補化（存在しないキーは自動で落とす）
			if self.primary_keys:
				key_candidates = []
				weight_candidates = []
				for i, k in enumerate(self.primary_keys):
					if k not in self._valid_primary_keys:
						warnings.warn(f'Unknown primary key "{k}" ignored.')
						continue
					if k == 'cmp' and not cmp_available:
						continue
					key_candidates.append(k)
					if self.primary_key_weights and i < len(self.primary_key_weights):
						weight_candidates.append(
							max(float(self.primary_key_weights[i]), 0.0)
						)
					else:
						weight_candidates.append(1.0)
				# すべて落ちた場合はデフォルトへフォールバック
				if not key_candidates:
					key_candidates = ['ffid', 'chno'] + (
						['cmp'] if cmp_available else []
					)
					weight_candidates = [1.0] * len(key_candidates)
			else:
				# 2) 従来デフォルト
				key_candidates = ['ffid', 'chno'] + (['cmp'] if cmp_available else [])
				weight_candidates = [1.0] * len(key_candidates)

			# 抽選（重みがあれば使用）
			if any(w > 0 for w in weight_candidates) and len(weight_candidates) == len(
				key_candidates
			):
				key_name = random.choices(
					key_candidates, weights=weight_candidates, k=1
				)[0]
			else:
				key_name = random.choice(key_candidates)
			unique_keys = info[f'{key_name}_unique_keys']
			key_to_indices = info[f'{key_name}_key_to_indices']
			if not unique_keys:
				continue
			key = random.choice(unique_keys)
			indices = key_to_indices[key]
			# === superwindow: collect neighboring primary keys (no averaging) ===
			do_super = self.use_superwindow and self.sw_halfspan > 0
			if do_super and self.sw_prob < 1.0:
				do_super = random.random() < self.sw_prob
			if do_super:
				# Build a window by position in the sorted list of unique primary keys
				uniq = info.get(f'{key_name}_unique_keys', None)
				if isinstance(uniq, (list, tuple)):
					uniq_arr = np.asarray(uniq, dtype=np.int64)
				else:
					uniq_arr = np.asarray([], dtype=np.int64)

				if uniq_arr.size > 0:
					uniq_sorted = np.sort(uniq_arr)
					center = int(key)
					pos = np.searchsorted(uniq_sorted, center)
					lo = max(0, pos - self.sw_halfspan)
					hi = min(len(uniq_sorted), pos + self.sw_halfspan + 1)
					win_keys = [int(k) for k in uniq_sorted[lo:hi]]
				else:
					win_keys = [int(key)]

				# Concatenate candidate indices from all keys in the window
				k2map = info[f'{key_name}_key_to_indices']
				chunks = []
				for k2 in win_keys:
					idxs = k2map.get(k2)
					if idxs is not None and len(idxs) > 0:
						chunks.append(idxs)
				if chunks:
					indices = np.concatenate(chunks).astype(np.int64)
				else:
					indices = np.asarray(indices, dtype=np.int64)
			else:
				indices = np.asarray(indices, dtype=np.int64)
			# === end superwindow ===

			# ---- secondary sort rules ----
			# 1st=FFID  -> 2nd=CHNO or OFFSET (random)
			# 1st=CHNO  -> 2nd=FFID or OFFSET (random)
			# 1st=CMP   -> 2nd=OFFSET
			try:
				prim_vals = info[f'{key_name}_values'][indices]

				# secondary を規則どおりに決める（FFID/CHNOはランダム分岐、CMPは固定）
				if key_name == 'ffid':
					secondary = random.choice(('chno', 'offset'))
				elif key_name == 'chno':
					secondary = random.choice(('ffid', 'offset'))
				else:  # key_name == 'cmp'
					secondary = 'offset'

				# secondary の値を取得
				if secondary == 'chno':
					sec_vals = info['chno_values'][indices]
				elif secondary == 'ffid':
					sec_vals = info['ffid_values'][indices]
				else:  # 'offset'
					sec_vals = info['offsets'][indices]

				# ★ secondary優先（列優先）にする安定ソート：
				# 先に primary、次に secondary を "mergesort"（安定）でかける
				o = np.argsort(prim_vals, kind='mergesort')
				indices = indices[o]
				sec_vals = sec_vals[o]

				o2 = np.argsort(sec_vals, kind='mergesort')
				indices = indices[o2]
			except Exception as e:
				print(f'Warning: secondary sort failed: {e}')
				print(f'  key_name={key_name}, indices.shape={indices.shape}')
				print(f'  prim_vals={prim_vals if "prim_vals" in locals() else "N/A"}')
				print(f'  sec_vals={sec_vals if "sec_vals" in locals() else "N/A"}')
				print(f'  {info["path"]}')
			# ---- end secondary sort ----

			n_total = len(indices)
			if n_total >= 128:
				start_idx = random.randint(0, n_total - 128)
				selected_indices = indices[start_idx : start_idx + 128]
				pad_len = 0
			else:
				selected_indices = indices
				pad_len = 128 - n_total
			selected_indices = np.asarray(selected_indices, dtype=np.int64)
			fb_subset = fb[selected_indices]
			if pad_len > 0:
				fb_subset = np.concatenate(
					[fb_subset, np.zeros(pad_len, dtype=fb_subset.dtype)]
				)
			offsets_full = info['offsets']
			off_subset = offsets_full[selected_indices].astype(np.float32)
			if pad_len > 0:
				off_subset = np.concatenate(
					[off_subset, np.zeros(pad_len, dtype=np.float32)]
				)
			pick_ratio = np.count_nonzero(fb_subset > 0) / len(fb_subset)
			if pick_ratio >= self.pick_ratio:
				break
		x = mmap[selected_indices].astype(np.float32)
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
			dt_eff_sec = info['dt_sec'] / max(factor, 1e-9)
			sample = {
				'masked': xm,
				'original': x_t,
				'fb_idx': fb_idx_t,
				'offsets': off_t,
				'dt_sec': torch.tensor(dt_eff_sec, dtype=torch.float32),
				'mask_indices': mask_idx,
				'key_name': key_name,
				'indices': selected_indices,
				'file_path': info['path'],
			}
			if self.target_mode == 'fb_seg':
				sample['target'] = target_t
			return sample

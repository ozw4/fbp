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

from proc.util.augment import (
	_apply_freq_augment,
	_spatial_stretch_sameH,
)
from proc.util.dataset.config import LoaderConfig
from proc.util.dataset.trace_subset_preproc import TraceSubsetLoader

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


def _build_centroids(key_to_indices, X, Y):
	if key_to_indices is None or X is None or Y is None:
		return None
	out = {}
	for k, idxs in key_to_indices.items():
		if idxs is None or len(idxs) == 0:
			continue
		# robust representative (median)
		mx = float(np.median(X[idxs]))
		my = float(np.median(Y[idxs]))
		out[int(k)] = (mx, my)
	return out


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
		reject_fblc: bool = False,
		fblc_percentile: float = 95.0,
		fblc_thresh_ms: float = 8.0,
		fblc_min_pairs: int = 16,
		fblc_apply_on: Literal['any', 'super_only'] = 'any',
		valid: bool = False,
		verbose: bool = False,
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
		self.reject_fblc = bool(reject_fblc)
		self.fblc_percentile = float(fblc_percentile)
		self.fblc_thresh_ms = float(fblc_thresh_ms)
		self.fblc_min_pairs = int(fblc_min_pairs)
		self.fblc_apply_on = fblc_apply_on
		self.valid = valid
		self.verbose = verbose
		self.subsetloader = TraceSubsetLoader(
			LoaderConfig(target_len=self.target_len, pad_traces_to=128)
		)
		self.file_infos = []
		for segy_path, fb_path in zip(self.segy_files, self.fb_files, strict=False):
			print(f'Loading {segy_path} and {fb_path}')
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

			# ---- distance centroids (FFID by source, CHNO by receiver) ----
			srcx = srcy = None
			grx = gry = None
			try:
				srcx = np.asarray(
					f.attributes(segyio.TraceField.SourceX)[:], dtype=np.float64
				)
				srcy = np.asarray(
					f.attributes(segyio.TraceField.SourceY)[:], dtype=np.float64
				)
			except Exception as e:
				warnings.warn(
					f'failed to read source coordinates from {segy_path}: {e}'
				)
				srcx = srcy = None
			try:
				grx = np.asarray(
					f.attributes(segyio.TraceField.GroupX)[:], dtype=np.float64
				)
				gry = np.asarray(
					f.attributes(segyio.TraceField.GroupY)[:], dtype=np.float64
				)
			except Exception as e:
				warnings.warn(
					f'failed to read receiver coordinates from {segy_path}: {e}'
				)
				grx = gry = None

			if (
				srcx is not None
				and srcy is not None
				and grx is not None
				and gry is not None
			):
				try:
					scal = np.asarray(
						f.attributes(segyio.TraceField.SourceGroupScalar)[:],
						dtype=np.float64,
					)
					scal_eff = np.where(
						scal == 0.0,
						1.0,
						np.where(scal > 0.0, scal, 1.0 / np.abs(scal)),
					)
					if scal_eff.size == 1 or scal_eff.size == srcx.size:
						srcx *= scal_eff
						srcy *= scal_eff
						grx *= scal_eff
						gry *= scal_eff
					else:
						warnings.warn(f'SourceGroupScalar size mismatch in {segy_path}')
						srcx = srcy = grx = gry = None
				except Exception as e:
					warnings.warn(
						f'failed to read SourceGroupScalar from {segy_path}: {e}'
					)
					srcx = srcy = grx = gry = None

			ffid_centroids = _build_centroids(ffid_key_to_indices, srcx, srcy)
			chno_centroids = _build_centroids(chno_key_to_indices, grx, gry)
			# ---------------------------------------------------------------

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
					ffid_centroids=ffid_centroids,
					chno_centroids=chno_centroids,
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
			secondary_key = 'none'
			info = random.choice(self.file_infos)
			mmap = info['mmap']
			fb = info['fb']

			cmp_available = (
				bool(info.get('cmp_unique_keys'))
				and isinstance(info['cmp_unique_keys'], (list, tuple))
				and len(info['cmp_unique_keys']) > 0
			)

			# ---- primary key selection (Hydra weights if provided) ----
			if self.primary_keys:
				key_candidates, weight_candidates = [], []
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
				if not key_candidates:  # fallback
					key_candidates = ['ffid', 'chno'] + (
						['cmp'] if cmp_available else []
					)
					weight_candidates = [1.0] * len(key_candidates)
			else:
				key_candidates = ['ffid', 'chno'] + (['cmp'] if cmp_available else [])
				weight_candidates = [1.0] * len(key_candidates)

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

			apply_super = False
			did_super = False  # ← このサンプルで実際に superwindow を適用したかどうか

			# === superwindow (distance-KNN for ffid/chno; index-window fallback) ===
			if self.use_superwindow and self.sw_halfspan > 0:
				apply_super = True
				if hasattr(self, 'sw_prob') and float(self.sw_prob) < 1.0:
					if random.random() >= float(self.sw_prob):
						apply_super = False

				if apply_super:
					did_super = True  # ← 実際に superwindow を使ったのでフラグON
					K = 1 + 2 * int(self.sw_halfspan)

					def _index_window():
						uniq = info.get(f'{key_name}_unique_keys', None)
						uniq_arr = (
							np.asarray(uniq, dtype=np.int64)
							if isinstance(uniq, (list, tuple))
							else np.asarray([], dtype=np.int64)
						)
						if uniq_arr.size > 0:
							uniq_sorted = np.sort(uniq_arr)
							center = int(key)
							pos = np.searchsorted(uniq_sorted, center)
							lo = max(0, pos - self.sw_halfspan)
							hi = min(len(uniq_sorted), pos + self.sw_halfspan + 1)
							return [int(k) for k in uniq_sorted[lo:hi]]
						return [int(key)]

					if key_name == 'ffid':
						cent = info.get('ffid_centroids', None)
						if isinstance(cent, dict) and int(key) in cent:
							keys = np.fromiter(cent.keys(), dtype=np.int64)
							coords = np.array(
								[cent[int(k)] for k in keys], dtype=np.float64
							)
							cx, cy = cent[int(key)]
							d = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
							order = np.argsort(d)
							sel_keys = keys[order][:K]
							k2map = info['ffid_key_to_indices']
						else:
							win_keys = _index_window()
							k2map = info[f'{key_name}_key_to_indices']
							sel_keys = np.asarray(win_keys, dtype=np.int64)
					elif key_name == 'chno':
						cent = info.get('chno_centroids', None)
						if isinstance(cent, dict) and int(key) in cent:
							keys = np.fromiter(cent.keys(), dtype=np.int64)
							coords = np.array(
								[cent[int(k)] for k in keys], dtype=np.float64
							)
							cx, cy = cent[int(key)]
							d = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
							order = np.argsort(d)
							sel_keys = keys[order][:K]
							k2map = info['chno_key_to_indices']
						else:
							win_keys = _index_window()
							k2map = info[f'{key_name}_key_to_indices']
							sel_keys = np.asarray(win_keys, dtype=np.int64)
					else:
						win_keys = _index_window()
						k2map = info[f'{key_name}_key_to_indices']
						sel_keys = np.asarray(win_keys, dtype=np.int64)

					chunks = []
					for k2 in sel_keys:
						idxs = k2map.get(int(k2))
						if idxs is not None and len(idxs) > 0:
							chunks.append(idxs)
					if chunks:
						indices = np.concatenate(chunks).astype(np.int64)
					else:
						indices = np.asarray(indices, dtype=np.int64)
				else:
					indices = np.asarray(indices, dtype=np.int64)
			else:
				indices = np.asarray(indices, dtype=np.int64)
			# === end superwindow ===

			# ---- secondary sort rules ----
			try:
				prim_vals = info[f'{key_name}_values'][indices]
				if not apply_super and not self.valid:
					if key_name == 'ffid':
						secondary = random.choice(('chno', 'offset'))
					elif key_name == 'chno':
						secondary = random.choice(('ffid', 'offset'))
					else:  # 'cmp'
						secondary = 'offset'
				elif apply_super and not self.valid:
					secondary = 'offset'
				elif self.valid:
					if key_name == 'ffid':
						secondary = 'chno'
					elif key_name == 'chno':
						secondary = 'ffid'
					else:  # 'cmp'
						secondary = 'offset'

				secondary_key = secondary
				if secondary == 'chno':
					sec_vals = info['chno_values'][indices]
				elif secondary == 'ffid':
					sec_vals = info['ffid_values'][indices]
				else:
					sec_vals = info['offsets'][indices]

				# stable lexicographic: primary then secondary
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

			# ---- take up to 128 traces (contiguous slice) ----
			n_total = len(indices)
			if n_total >= 128:
				start_idx = random.randint(0, n_total - 128)
				selected_indices = indices[start_idx : start_idx + 128]
				pad_len = 0
			else:
				selected_indices = indices
				pad_len = 128 - n_total
			selected_indices = np.asarray(selected_indices, dtype=np.int64)

			# primary unique set (for logging)
			prim_vals_sel = info[f'{key_name}_values'][selected_indices].astype(
				np.int64
			)
			primary_label_values = np.unique(prim_vals_sel)
			primary_unique_str = ','.join(map(str, primary_label_values.tolist()))

			# picks / offsets
			fb_subset = fb[selected_indices]
			if pad_len > 0:
				fb_subset = np.concatenate(
					[fb_subset, np.zeros(pad_len, dtype=fb_subset.dtype)]
				)
			off_subset = info['offsets'][selected_indices].astype(np.float32)
			if pad_len > 0:
				off_subset = np.concatenate(
					[off_subset, np.zeros(pad_len, dtype=np.float32)]
				)

			# require enough picks
			pick_ratio = np.count_nonzero(fb_subset > 0) / len(fb_subset)
			if pick_ratio < self.pick_ratio:
				continue  # retry whole sample

			# ---- load traces, normalize, augment ----
			x = self.subsetloader.load_and_normalize(mmap, selected_indices)

			# optional flip
			if self.flip and random.random() < 0.5:
				x = np.flip(x, axis=0).copy()
				fb_subset = fb_subset[::-1].copy()
				off_subset = off_subset[::-1].copy()

			# time augment
			factor = 1.0
			if self.augment_time_prob > 0 and random.random() < self.augment_time_prob:
				factor = random.uniform(*self.augment_time_range)
				frac = Fraction(factor).limit_denominator(128)
				up, down = frac.numerator, frac.denominator
				H_tmp = x.shape[0]
				x = np.stack(
					[
						resample_poly(x[h], up, down, padtype='line')
						for h in range(H_tmp)
					],
					axis=0,
				)

			# fit/crop/pad time length
			x, start = self._fit_time_len(x)

			# space augment (and keep offsets in sync)
			did_space = False
			f_h = 1.0
			if (
				self.augment_space_prob > 0
				and random.random() < self.augment_space_prob
			):
				f_h = random.uniform(*self.augment_space_range)
				x = _spatial_stretch_sameH(x, f_h)
				off_subset = _spatial_stretch_sameH(off_subset[:, None], f_h)[
					:, 0
				].astype(np.float32)
				did_space = True

			# freq augment
			if self.augment_freq_prob > 0 and random.random() < self.augment_freq_prob:
				x = _apply_freq_augment(
					x,
					self.augment_freq_kinds,
					self.augment_freq_band,
					self.augment_freq_width,
					self.augment_freq_roll,
					self.augment_freq_restandardize,
				)

			# first-break indices in window
			fb_idx_win = np.floor(fb_subset * factor).astype(np.int64) - start
			invalid = (fb_idx_win <= 0) | (fb_idx_win >= self.target_len)
			fb_idx_win[invalid] = -1

			# FBLC gate (before masking/target/tensorization)
			dt_eff_sec = info['dt_sec'] / max(factor, 1e-9)
			if self.reject_fblc:
				# super_only の場合は「実際に superwindow を使ったサンプル」に限って判定
				if self.fblc_apply_on == 'any':
					apply_gate = True
				elif self.fblc_apply_on == 'super_only':
					apply_gate = did_super
				else:
					apply_gate = False

				if apply_gate:
					v = fb_idx_win.astype(np.float64)
					valid = v >= 0
					m = valid[1:] & valid[:-1]
					valid_pairs = int(m.sum())

					# enough valid neighbor pairs?
					if valid_pairs < int(self.fblc_min_pairs):
						continue  # サンプル棄却して再抽選

					diffs = np.abs(v[1:] - v[:-1])[m]  # samples
					p = float(np.percentile(diffs, float(self.fblc_percentile)))
					p_ms = p * float(dt_eff_sec) * 1000.0
					if p_ms > float(self.fblc_thresh_ms):
						if self.verbose:
							print(
								f'Rejecting gather {info["path"]} key={key_name}:{key} '
								f'for FBLC {p_ms:.1f}ms > {self.fblc_thresh_ms}ms'
							)
						continue  # reject and resample
					if self.verbose:
						print(
							f'Accepted gather {info["path"]} key={key_name}:{key} '
							f'for FBLC {p_ms:.1f}ms <= {self.fblc_thresh_ms}ms'
						)

			# masking (after acceptance)
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

			# target (optional)
			target_t = None
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
				if did_space:
					target = _spatial_stretch_sameH(target, f_h)
				target_t = torch.from_numpy(target)[None, ...]

			# tensors + sample dict
			x_t = torch.from_numpy(x)[None, ...]
			xm = torch.from_numpy(x_masked)[None, ...]
			fb_idx_t = torch.from_numpy(fb_idx_win)
			off_t = torch.from_numpy(off_subset)

			sample = {
				'masked': xm,
				'original': x_t,
				'fb_idx': fb_idx_t,
				'offsets': off_t,
				'dt_sec': torch.tensor(dt_eff_sec, dtype=torch.float32),
				'mask_indices': mask_idx,
				'key_name': key_name,
				'secondary_key': secondary_key,
				'indices': selected_indices,
				'file_path': info['path'],
				'primary_unique': primary_unique_str,
			}
			if self.verbose:
				sample['did_superwindow'] = did_super
				print(f'primary_key: {key_name}={primary_unique_str}')
				print(f'secondary_key: {secondary_key}')
			if target_t is not None:
				sample['target'] = target_t

			return sample

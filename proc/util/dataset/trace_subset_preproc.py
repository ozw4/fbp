from __future__ import annotations

import numpy as np

from .config import LoaderConfig
from .signal_util import standardize_per_trace


class TraceSubsetLoader:
	def __init__(self, cfg: LoaderConfig):
		if cfg.target_len <= 0:
			raise ValueError('target_len must be positive')
		if cfg.pad_traces_to <= 0:
			raise ValueError('pad_traces_to must be positive')
		self.cfg = cfg

	# ---- split steps ----
	def load_traces(self, mmap, indices: np.ndarray) -> np.ndarray:
		idx = np.asarray(indices, dtype=np.int64)
		x = mmap[idx].astype(np.float32)  # (H0, T)
		return x

	def pad_traces_to_H(self, x: np.ndarray) -> np.ndarray:
		H0, T = x.shape
		if self.cfg.pad_traces_to <= H0:
			return x
		pad = self.cfg.pad_traces_to - H0
		z = np.zeros((pad, T), dtype=x.dtype)
		return np.concatenate([x, z], axis=0)

	def fit_time_len(
		self,
		x: np.ndarray,
		start: int | None = None,
		rng: np.random.Generator | None = None,
	) -> tuple[np.ndarray, int]:
		if x.ndim != 2:
			raise ValueError(f'x must be 2D (H,T), got {x.shape}')
		H, T = x.shape
		target = self.cfg.target_len

		if target < T:
			if start is None:
				start = (
					np.random.randint(0, T - target + 1)
					if rng is None
					else int(rng.integers(0, T - target + 1))
				)
			if not (0 <= start <= T - target):
				raise ValueError(f'invalid start={start} for T={T}, target={target}')
			return x[:, start : start + target], start

		if target > T:
			pad = target - T
			x_pad = np.pad(x, ((0, 0), (0, pad)), mode='constant')
			return x_pad, (start or 0)

		return x, (start or 0)

	# ---- convenience (keeps old call-sites simple) ----
	def load_and_normalize(self, mmap, indices: np.ndarray) -> np.ndarray:
		x = self.load_traces(mmap, indices)
		x = self.pad_traces_to_H(x)
		x = standardize_per_trace(x)
		return x

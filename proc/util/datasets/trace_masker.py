from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

MaskMode = Literal['replace', 'add']


@dataclass(frozen=True)
class TraceMaskerConfig:
	mask_ratio: float = 0.5  # [0,1]
	mode: MaskMode = 'replace'  # "replace" or "add"
	noise_std: float = 1.0  # Gaussian std


class TraceMasker:
	"""Apply masking to an (H, T) float32 array x.
	Picks int(mask_ratio*H) trace indices via random.sample,
	and either REPLACE those traces with noise or ADD noise.
	Returns (x_masked, mask_indices).
	"""

	def __init__(self, cfg: TraceMaskerConfig):
		if not (0.0 <= cfg.mask_ratio <= 1.0):
			raise ValueError('mask_ratio must be in [0,1]')
		if cfg.mode not in ('replace', 'add'):
			raise ValueError(f'invalid mode: {cfg.mode}')
		if cfg.noise_std < 0:
			raise ValueError('noise_std must be >= 0')
		self.cfg = cfg

	def apply(
		self,
		x: np.ndarray,
		*,
		mask_ratio: float | None = None,
		mode: Literal['replace', 'add'] | None = None,
		noise_std: float | None = None,
		py_random: random.Random | None = None,
	) -> tuple[np.ndarray, list[int]]:
		# 1) resolve current params
		ratio = self.cfg.mask_ratio if mask_ratio is None else float(mask_ratio)
		md = self.cfg.mode if mode is None else mode
		std = self.cfg.noise_std if noise_std is None else float(noise_std)

		# 2) validate
		if not (0.0 <= ratio <= 1.0):
			raise ValueError('mask_ratio must be in [0,1]')
		if md not in ('replace', 'add'):
			raise ValueError(f'invalid mode: {md}')
		if std < 0:
			raise ValueError('noise_std must be >= 0')
		if x.ndim != 2:
			raise ValueError(f'x must be 2D (H,T), got {x.shape}')

		# 3) mask computation using the *resolved* params
		H, T = x.shape
		num_mask = int(ratio * H)
		if num_mask <= 0:
			return x.copy(), []  # keep original but return a copy

		r = py_random or random
		mask_idx = r.sample(range(H), num_mask)  # legacy-compatible sampling

		x_masked = x.copy()
		noise = np.random.normal(0.0, std, size=(num_mask, T)).astype(np.float32)
		if md == 'replace':
			x_masked[mask_idx] = noise
		else:  # 'add'
			x_masked[mask_idx] += noise

		return x_masked, mask_idx

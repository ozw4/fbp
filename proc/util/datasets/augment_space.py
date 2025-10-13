from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from proc.util.augment import _spatial_stretch_sameH


@dataclass(frozen=True)
class SpaceAugConfig:
	prob: float = 0.0
	range: tuple[float, float] = (0.90, 1.10)


class SpaceAugmenter:
	def __init__(self, config: SpaceAugConfig | None = None) -> None:
		self.config = config if config is not None else SpaceAugConfig()

	def apply(
		self,
		x: np.ndarray,
		offsets: np.ndarray,
		*,
		rng_py: random.Random | None = None,
		prob: float | None = None,
		range: tuple[float, float] | None = None,
	) -> tuple[np.ndarray, np.ndarray, bool, float]:
		rng = rng_py if rng_py is not None else random
		prob_eff = self.config.prob if prob is None else prob
		if prob_eff <= 0.0 or rng.random() >= prob_eff:
			return x, offsets, False, 1.0

		range_eff = self.config.range if range is None else range
		f_h = rng.uniform(range_eff[0], range_eff[1])

		x_aug = _spatial_stretch_sameH(x, f_h)
		offsets_aug = _spatial_stretch_sameH(offsets[:, None], f_h)[:, 0]
		return (
			x_aug.astype(np.float32, copy=False),
			offsets_aug.astype(np.float32, copy=False),
			True,
			f_h,
		)

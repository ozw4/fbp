from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from proc.util.augment import _apply_freq_augment


@dataclass(frozen=True)
class FreqAugConfig:
	prob: float = 0.0
	kinds: tuple[str, ...] = ('bandpass', 'lowpass', 'highpass')
	band: tuple[float, float] = (0.05, 0.45)
	width: tuple[float, float] = (0.10, 0.35)
	roll: float = 0.02
	restandardize: bool = True


class FreqAugmenter:
	def __init__(self, config: FreqAugConfig | None = None) -> None:
		self.config = config if config is not None else FreqAugConfig()

	def apply(
		self,
		x: np.ndarray,
		*,
		rng_py: random.Random | None = None,
		prob: float | None = None,
		kinds: tuple[str, ...] | None = None,
		band: tuple[float, float] | None = None,
		width: tuple[float, float] | None = None,
		roll: float | None = None,
		restandardize: bool | None = None,
	) -> np.ndarray:
		rng = rng_py if rng_py is not None else random
		prob_eff = self.config.prob if prob is None else prob
		if prob_eff <= 0.0 or rng.random() >= prob_eff:
			return x

		kinds_eff = self.config.kinds if kinds is None else kinds
		band_eff = self.config.band if band is None else band
		width_eff = self.config.width if width is None else width
		roll_eff = self.config.roll if roll is None else roll
		restandardize_eff = (
			self.config.restandardize if restandardize is None else restandardize
		)

		x_aug = _apply_freq_augment(
			x,
			kinds_eff,
			band_eff,
			width_eff,
			roll_eff,
			restandardize_eff,
		)
		return np.asarray(x_aug, dtype=np.float32)

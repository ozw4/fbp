from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import random
import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class TimeAugConfig:
        prob: float = 0.0
        range: tuple[float, float] = (0.95, 1.05)


class TimeAugmenter:
        def __init__(self, config: TimeAugConfig | None = None) -> None:
                self.config = config if config is not None else TimeAugConfig()

        def apply(
                self,
                x: np.ndarray,
                *,
                rng_py: random.Random | None = None,
                prob: float | None = None,
                range: tuple[float, float] | None = None,
        ) -> tuple[np.ndarray, float]:
                rng = rng_py if rng_py is not None else random
                prob_eff = self.config.prob if prob is None else prob
                if prob_eff <= 0.0 or rng.random() >= prob_eff:
                        return x, 1.0

                range_eff = self.config.range if range is None else range
                factor = rng.uniform(range_eff[0], range_eff[1])
                frac = Fraction(factor).limit_denominator(128)
                up, down = frac.numerator, frac.denominator

                traces = [
                        resample_poly(trace, up, down, padtype='line') for trace in np.asarray(x)
                ]
                x_aug = np.stack(traces, axis=0).astype(np.float32, copy=False)
                return x_aug, factor

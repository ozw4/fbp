from dataclasses import dataclass

import numpy as np

from proc.util.augment import _spatial_stretch_sameH


@dataclass(frozen=True)
class FBTargetConfig:
    sigma: float = 1.0


class FBTargetBuilder:
    def __init__(self, cfg: FBTargetConfig) -> None:
        self._cfg = cfg

    def build(
        self,
        fb_idx_win: np.ndarray,
        W: int,
        *,
        did_space: bool = False,
        f_h: float = 1.0,
        sigma: float | None = None,
    ) -> np.ndarray:
        sigma_in = self._cfg.sigma if sigma is None else sigma
        sigma_eff = max(float(sigma_in), 1e-6)
        target = np.zeros((fb_idx_win.shape[0], W), dtype=np.float32)
        valid = fb_idx_win >= 0
        if np.any(valid):
            t = np.arange(W, dtype=np.float32)[None, :]
            idxv = fb_idx_win[valid].astype(np.float32)[:, None]
            g = np.exp(-0.5 * ((t - idxv) / sigma_eff) ** 2)
            g /= g.max(axis=1, keepdims=True) + 1e-12
            target[valid] = g.astype(np.float32)
        if did_space:
            target = _spatial_stretch_sameH(target, f_h)
        target = target.astype(np.float32, copy=False)
        return target[None, ...]

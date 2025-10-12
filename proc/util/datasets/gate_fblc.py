from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FirstBreakGateConfig:
        percentile: float = 95.0
        thresh_ms: float = 8.0
        min_pairs: int = 16
        apply_on: Literal['any', 'super_only'] = 'any'
        verbose: bool = False


class FirstBreakGate:
        def __init__(self, cfg: FirstBreakGateConfig):
                if not (0.0 < float(cfg.percentile) < 100.0):
                        raise ValueError('percentile must be in (0, 100)')
                if not (float(cfg.thresh_ms) > 0.0):
                        raise ValueError('thresh_ms must be positive')
                if int(cfg.min_pairs) < 0:
                        raise ValueError('min_pairs must be non-negative')
                if cfg.apply_on not in ('any', 'super_only'):
                        raise ValueError("apply_on must be 'any' or 'super_only'")
                self.cfg = cfg

        def should_apply(
                self,
                *,
                did_super: bool,
                apply_on: Literal['any', 'super_only'] | None = None,
        ) -> bool:
                ap = self.cfg.apply_on if apply_on is None else apply_on
                if ap == 'any':
                        return True
                if ap == 'super_only':
                        return bool(did_super)
                return False

        def accept(
                self,
                fb_idx_win: np.ndarray,
                dt_eff_sec: float,
                *,
                did_super: bool,
                percentile: float | None = None,
                thresh_ms: float | None = None,
                min_pairs: int | None = None,
                apply_on: Literal['any', 'super_only'] | None = None,
        ) -> tuple[bool, float | None, int]:
                p = self.cfg.percentile if percentile is None else float(percentile)
                th = self.cfg.thresh_ms if thresh_ms is None else float(thresh_ms)
                mp = self.cfg.min_pairs if min_pairs is None else int(min_pairs)
                ap = self.cfg.apply_on if apply_on is None else apply_on

                if not (0.0 < float(p) < 100.0):
                        raise ValueError('percentile must be in (0, 100)')
                if not (float(th) > 0.0):
                        raise ValueError('thresh_ms must be positive')
                if int(mp) < 0:
                        raise ValueError('min_pairs must be non-negative')
                if ap not in ('any', 'super_only'):
                        raise ValueError("apply_on must be 'any' or 'super_only'")

                if not self.should_apply(did_super=did_super, apply_on=ap):
                        return True, None, 0

                v = fb_idx_win.astype(np.float64)
                valid = v >= 0
                m = valid[1:] & valid[:-1]
                valid_pairs = int(m.sum())

                if valid_pairs < int(mp):
                        return False, None, valid_pairs

                diffs = np.abs(v[1:] - v[:-1])[m]
                q = float(np.percentile(diffs, float(p)))
                p_ms = q * float(dt_eff_sec) * 1000.0
                ok = p_ms <= float(th)
                return ok, p_ms, valid_pairs

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

        def should_apply(self, *, did_super: bool) -> bool:
                if self.cfg.apply_on == 'any':
                        return True
                if self.cfg.apply_on == 'super_only':
                        return bool(did_super)
                return False

        def accept(
                self,
                fb_idx_win: np.ndarray,
                dt_eff_sec: float,
                *,
                did_super: bool,
        ) -> tuple[bool, float | None, int]:
                if not self.should_apply(did_super=did_super):
                        return True, None, 0

                v = fb_idx_win.astype(np.float64)
                valid = v >= 0
                m = valid[1:] & valid[:-1]
                valid_pairs = int(m.sum())

                if valid_pairs < int(self.cfg.min_pairs):
                        return False, None, valid_pairs

                diffs = np.abs(v[1:] - v[:-1])[m]
                p = float(np.percentile(diffs, float(self.cfg.percentile)))
                p_ms = p * float(dt_eff_sec) * 1000.0
                ok = p_ms <= float(self.cfg.thresh_ms)
                return ok, p_ms, valid_pairs

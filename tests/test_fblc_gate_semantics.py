import numpy as np

from proc.util.datasets.gate_fblc import FirstBreakGate, FirstBreakGateConfig


def test_fblc_accept_any():
        fb = np.array([10, 11, 10, 11, 10, 11, 10, 11, 10, 11], dtype=np.int64)
        gate = FirstBreakGate(
                FirstBreakGateConfig(percentile=95.0, thresh_ms=8.0, min_pairs=4, apply_on="any")
        )
        ok, p_ms, pairs = gate.accept(fb, dt_eff_sec=0.001, did_super=False)
        assert pairs >= 4 and ok and (p_ms is not None)


def test_fblc_reject_large_jitter():
        fb = np.array([10, 50, 10, 50, 10, 50, 10, 50, 10, 50], dtype=np.int64)
        gate = FirstBreakGate(
                FirstBreakGateConfig(percentile=95.0, thresh_ms=1.0, min_pairs=4, apply_on="any")
        )
        ok, p_ms, pairs = gate.accept(fb, dt_eff_sec=0.002, did_super=True)
        assert pairs >= 4 and (p_ms is not None) and (not ok)


def test_fblc_super_only_bypass_when_not_applied():
        fb = np.array([-1, -1, 10, 10, 10, 10, -1, -1], dtype=np.int64)
        gate = FirstBreakGate(FirstBreakGateConfig(apply_on="super_only"))
        ok, p_ms, pairs = gate.accept(fb, dt_eff_sec=0.001, did_super=False)
        assert ok and p_ms is None

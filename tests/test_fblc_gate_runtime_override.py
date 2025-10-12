import numpy as np

from proc.util.datasets.gate_fblc import FirstBreakGate, FirstBreakGateConfig


def test_runtime_override_thresh_ms_changes_decision():
        fb = np.array([0, 20, 0, 20, 0, 20, 0, 20], dtype=np.int64)
        gate = FirstBreakGate(
                FirstBreakGateConfig(percentile=95.0, thresh_ms=1.0, min_pairs=3, apply_on="any")
        )
        ok1, p_ms1, pairs1 = gate.accept(fb, dt_eff_sec=0.001, did_super=False)
        assert pairs1 >= 3 and (p_ms1 is not None) and (ok1 is False)

        ok2, p_ms2, pairs2 = gate.accept(
                fb,
                dt_eff_sec=0.001,
                did_super=False,
                thresh_ms=10_000.0,
        )
        assert pairs2 == pairs1 and (p_ms2 is not None) and (ok2 is True)


def test_runtime_override_apply_on_bypass_vs_force_apply():
        fb = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
        gate = FirstBreakGate(FirstBreakGateConfig(apply_on="super_only", min_pairs=2, thresh_ms=1.0))
        ok_bypass, p_ms_bypass, pairs_bypass = gate.accept(fb, dt_eff_sec=0.001, did_super=False)
        assert ok_bypass is True and p_ms_bypass is None and pairs_bypass == 0

        ok_force, p_ms_force, pairs_force = gate.accept(
                fb,
                dt_eff_sec=0.001,
                did_super=False,
                apply_on="any",
        )
        assert p_ms_force is not None and pairs_force > 0

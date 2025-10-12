import numpy as np

from proc.util.datasets.target_fb import FBTargetConfig, FBTargetBuilder


def _peak_width_at_half_maximum(row):
    return int((row > 0.5).sum())


def test_target_fb_sigma_override_changes_width():
    H, W = 1, 200
    fb = np.array([100], dtype=np.int64)
    builder = FBTargetBuilder(FBTargetConfig(sigma=2.0))
    y1 = builder.build(fb, W)
    y2 = builder.build(fb, W, sigma=8.0)
    w1 = _peak_width_at_half_maximum(y1[0, 0])
    w2 = _peak_width_at_half_maximum(y2[0, 0])
    assert w2 > w1

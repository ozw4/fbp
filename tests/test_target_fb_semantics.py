import numpy as np

from proc.util.datasets.target_fb import FBTargetConfig, FBTargetBuilder


def test_target_fb_semantics_basic():
    H, W = 8, 64
    fb = np.full(H, -1, dtype=np.int64)
    fb[::2] = 10
    builder = FBTargetBuilder(FBTargetConfig(sigma=2.0))
    y = builder.build(fb, W)
    assert y.shape == (1, H, W)
    assert y.dtype == np.float32
    mx = y[0].max(axis=1)
    assert np.allclose(mx[::2], 1.0, atol=1e-6)
    assert np.allclose(y[0, 1::2], 0.0)

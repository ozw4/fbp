import numpy as np

from proc.util.datasets.trace_masker import TraceMasker, TraceMaskerConfig


def test_trace_masker_semantics_replace():
        H, T = 8, 16
        x = np.random.randn(H, T).astype(np.float32)
        m = TraceMasker(TraceMaskerConfig(mask_ratio=0.25, mode="replace", noise_std=1.0))
        xm, idx = m.apply(x)
        assert xm.shape == x.shape and xm.dtype == np.float32
        assert len(idx) == int(0.25 * H)
        keep = np.ones(H, dtype=bool)
        keep[idx] = False
        assert np.allclose(xm[keep], x[keep])  # untouched rows are identical


def test_trace_masker_semantics_add():
        H, T = 8, 16
        x = np.zeros((H, T), dtype=np.float32)
        m = TraceMasker(TraceMaskerConfig(mask_ratio=0.5, mode="add", noise_std=0.5))
        xm, idx = m.apply(x)
        assert xm.shape == x.shape
        # masked rows should be non-zero with high probability
        assert (np.abs(xm[idx]).mean() > 0)

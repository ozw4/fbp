import numpy as np

from proc.util.datasets.augment_space import SpaceAugConfig, SpaceAugmenter


def test_space_aug_noop_when_prob_zero():
	H, W = 8, 32
	x = np.random.randn(H, W).astype(np.float32)
	off = np.linspace(0, 1, H).astype(np.float32)
	aug = SpaceAugmenter(SpaceAugConfig(prob=0.0))
	y, off2, did, f = aug.apply(x, off, rng_py=None)
	assert did is False and f == 1.0
	np.testing.assert_allclose(y, x, atol=0, rtol=0)
	np.testing.assert_allclose(off2, off, atol=0, rtol=0)

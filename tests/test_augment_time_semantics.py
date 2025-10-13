import numpy as np

from proc.util.datasets.augment_time import TimeAugConfig, TimeAugmenter


def test_time_aug_noop_when_prob_zero():
	H, W = 4, 64
	x = np.random.randn(H, W).astype(np.float32)
	aug = TimeAugmenter(TimeAugConfig(prob=0.0))
	y, factor = aug.apply(x, rng_py=None)
	assert factor == 1.0
	np.testing.assert_allclose(y, x, atol=0, rtol=0)

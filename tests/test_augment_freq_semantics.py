import numpy as np

from proc.util.datasets.augment_freq import FreqAugConfig, FreqAugmenter


def test_freq_aug_noop_when_prob_zero():
	H, W = 8, 64
	x = np.random.randn(H, W).astype(np.float32)
	aug = FreqAugmenter(FreqAugConfig(prob=0.0))
	y = aug.apply(x, rng_py=None)
	np.testing.assert_allclose(y, x, atol=0, rtol=0)

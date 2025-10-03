import numpy as np


def standardize_per_trace(x: np.ndarray) -> np.ndarray:
	# per-trace (row-wise) zero-mean, unit-std with small epsilon
	m = x.mean(axis=1, keepdims=True)
	s = x.std(axis=1, keepdims=True) + 1e-10
	return (x - m) / s

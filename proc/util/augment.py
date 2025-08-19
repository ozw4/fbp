import random
from fractions import Fraction

import numpy as np
from scipy.ndimage import zoom as nd_zoom
from scipy.signal import resample_poly

__all__ = [
	'_apply_freq_augment',
	'_cosine_ramp',
	'_fit_time_len_np',
	'_make_freq_mask',
	'_spatial_stretch_sameH',
	'_time_stretch_poly',
]

def _time_stretch_poly(x_hw: np.ndarray, factor: float, target_len: int) -> np.ndarray:
	"""Stretch (H,W) array in time and fit to target length."""
	if abs(factor - 1.0) < 1e-4:
		return _fit_time_len_np(x_hw, target_len)
	H, W = x_hw.shape
	frac = Fraction(factor).limit_denominator(128)
	up, down = frac.numerator, frac.denominator
	y = np.stack([
		resample_poly(x_hw[h], up, down, padtype='line') for h in range(H)
	], axis=0)
	return _fit_time_len_np(y, target_len)

def _fit_time_len_np(x_hw: np.ndarray, target_len: int) -> np.ndarray:
	"""Trim or pad (H,W') to target_len along time axis."""
	W = x_hw.shape[1]
	if target_len == W:
		return x_hw
	if target_len < W:
		start = np.random.randint(0, W - target_len + 1)
		return x_hw[:, start : start + target_len]
	pad = target_len - W
	return np.pad(x_hw, ((0, 0), (0, pad)), mode='constant')

def _spatial_stretch_sameH(x_hw: np.ndarray, factor: float) -> np.ndarray:
	"""Stretch traces spatially while keeping original count."""
	if abs(factor - 1.0) < 1e-4:
		return x_hw
	H, W = x_hw.shape
	H2 = max(1, int(round(H * factor)))
	y = nd_zoom(x_hw, zoom=(H2 / H, 1.0), order=1, mode='reflect', prefilter=False)
	y = nd_zoom(y, zoom=(H / H2, 1.0), order=1, mode='reflect', prefilter=False)
	if y.shape[0] < H:
		y = np.pad(y, ((0, H - y.shape[0]), (0, 0)), mode='edge')
	elif y.shape[0] > H:
		y = y[:H, :]
	return y

def _cosine_ramp(x: np.ndarray, a: float, b: float, invert: bool = False) -> np.ndarray:
	"""Cosine ramp from 0 to 1 (or inverted) over [a, b]."""
	if b <= a:
		return np.ones_like(x) if not invert else np.zeros_like(x)
	t = np.clip((x - a) / (b - a), 0.0, 1.0)
	ramp = 0.5 - 0.5 * np.cos(np.pi * t)
	return (1.0 - ramp) if invert else ramp

def _make_freq_mask(
	n_rfft: int,
	kind: str,
	f_lo: float | None,
	f_hi: float | None,
	roll: float,
) -> np.ndarray:
	"""Create smooth frequency mask for rFFT."""
	f = np.linspace(0.0, 1.0, n_rfft, dtype=np.float32)
	m = np.ones_like(f, dtype=np.float32)
	if kind == 'bandpass':
		assert f_lo is not None and f_hi is not None and f_hi > f_lo
		bw = max(f_hi - f_lo, 1e-4)
		r = max(roll, 0.05 * bw)
		up = _cosine_ramp(f, max(0.0, f_lo - r), min(1.0, f_lo + r), invert=False)
		dn = _cosine_ramp(f, max(0.0, f_hi - r), min(1.0, f_hi + r), invert=True)
		m = up * dn
	elif kind == 'lowpass':
		assert f_hi is not None
		r = max(roll, 0.05 * f_hi)
		dn = _cosine_ramp(f, max(0.0, f_hi - r), min(1.0, f_hi + r), invert=True)
		m = dn
	elif kind == 'highpass':
		assert f_lo is not None
		r = max(roll, 0.05 * (1.0 - f_lo))
		up = _cosine_ramp(f, max(0.0, f_lo - r), min(1.0, f_lo + r), invert=False)
		m = up
	else:
		raise ValueError(f'unknown freq-augment kind: {kind}')
	return m.astype(np.float32)

def _apply_freq_augment(
	x_hw: np.ndarray,
	augment_freq_kinds: tuple[str, ...],
	augment_freq_band: tuple[float, float],
	augment_freq_width: tuple[float, float],
	augment_freq_roll: float,
	augment_freq_restandardize: bool,
) -> np.ndarray:
	"""Apply same frequency mask to all traces."""
	H, W = x_hw.shape
	kind = random.choice(augment_freq_kinds)
	lo_min, hi_max = augment_freq_band
	if kind == 'bandpass':
		min_w, max_w = augment_freq_width
		bw = random.uniform(min_w, max_w)
		f_lo = random.uniform(lo_min, max(1e-3, hi_max - bw))
		f_hi = min(hi_max, f_lo + bw)
	elif kind == 'lowpass':
		f_lo, f_hi = None, random.uniform(lo_min + 0.02, hi_max)
	elif kind == 'highpass':
		f_lo, f_hi = random.uniform(lo_min, hi_max - 0.02), None
	else:
		raise ValueError(f'unknown freq-augment kind: {kind}')
	n_rfft = W // 2 + 1
	mask = _make_freq_mask(n_rfft, kind, f_lo, f_hi, augment_freq_roll)
	X = np.fft.rfft(x_hw, axis=1)
	Y = X * mask[None, :]
	y = np.fft.irfft(Y, n=W, axis=1).astype(np.float32)
	if augment_freq_restandardize:
		m = y.mean(axis=1, keepdims=True)
		s = y.std(axis=1, keepdims=True) + 1e-10
		y = (y - m) / s
	return y

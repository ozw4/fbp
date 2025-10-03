# %%
import random

import numpy as np
import torch

from proc.util.datasets.masked_segy_gather import MaskedSegyGather as New
from proc.util.datasets.masked_segy_gather_legacy import MaskedSegyGather as Legacy


def to_np(x):
	if isinstance(x, torch.Tensor):
		return x.detach().cpu().numpy()
	return np.asarray(x)


def build_args():
	# 実データに合わせて
	common = dict(
		segy_files=[
			'/home/dcuser/data/ActiveSeisField/aso19-1/input_TRCTAB_ml_fbpick_Aso19-1_wolmo.sgy'
		],
		fb_files=['/home/dcuser/data/ActiveSeisField/aso19-1/fb_Aso19-1.npy'],
		use_superwindow=False,  # 決定論性を上げる
		augment_time_prob=0.0,
		augment_space_prob=0.0,
		augment_freq_prob=0.0,
		valid=True,  # secondary sort を固定化
		verbose=False,
	)
	return common


def main():
	args = build_args()
	ds_old = Legacy(**args)
	ds_new = New(**args)

	for i in range(50):
		# 毎回同じ乱数系列にする（2データセットが同じ分岐を通るように）
		random.seed(i)
		np.random.seed(i)
		torch.manual_seed(i)

		a = ds_old[None]
		random.seed(i)
		np.random.seed(i)
		torch.manual_seed(i)
		b = ds_new[None]

		assert a.keys() == b.keys()

		def same(name, atol=1e-6, rtol=1e-6):
			xa, xb = a[name], b[name]
			if isinstance(xa, (np.ndarray, torch.Tensor)) or isinstance(
				xb, (np.ndarray, torch.Tensor)
			):
				aa, bb = to_np(xa), to_np(xb)
				assert aa.shape == bb.shape, f'{name}: shape {aa.shape}!={bb.shape}'
				np.testing.assert_allclose(aa, bb, atol=atol, rtol=rtol)
			else:
				assert xa == xb, f'{name}: {xa}!={xb}'

		# 必須キーの一致（必要に応じて増やす）
		for k in [
			'masked',
			'original',
			'fb_idx',
			'offsets',
			'dt_sec',
			'key_name',
			'secondary_key',
			'primary_unique',
		]:
			same(k)

	print('OK: legacy == new')


if __name__ == '__main__':
	main()

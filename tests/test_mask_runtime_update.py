import os

import numpy as np
import pytest
import torch

from proc.util.datasets.masked_segy_gather import MaskedSegyGather

SEGY = os.getenv('FBP_TEST_SEGY')  # 例: /path/to/data.sgy
FBNP = os.getenv('FBP_TEST_FB')  # 例: /path/to/data_fb.npy

pytestmark = pytest.mark.skipif(
	not (SEGY and FBNP),
	reason='Set FBP_TEST_SEGY and FBP_TEST_FB to run this test.',
)


def _build_ds(mask_ratio: float) -> MaskedSegyGather:
	return MaskedSegyGather(
		segy_files=[SEGY],
		fb_files=[FBNP],
		# ランダム性・再抽選を抑える（仕様テスト向け）
		use_superwindow=False,
		augment_time_prob=0.0,
		augment_space_prob=0.0,
		augment_freq_prob=0.0,
		flip=False,
		pick_ratio=0.0,
		reject_fblc=False,
		valid=True,
		verbose=False,
		mask_ratio=mask_ratio,
		mask_mode='replace',
		mask_noise_std=1.0,
	)


def test_mask_ratio_runtime_update_changes_mask_count():
	ds = _build_ds(mask_ratio=0.0)

	# 1st: マスク無効
	a = ds[None]
	Ha = a['original'].shape[1]
	assert isinstance(a['masked'], torch.Tensor)
	assert len(a['mask_indices']) == 0

	# 2nd: ランタイムに 50% へ更新 → 本数が変わる
	ds.mask_ratio = 0.5
	b = ds[None]
	Hb = b['original'].shape[1]
	assert len(b['mask_indices']) == int(0.5 * Hb)

	ds.close()


def test_mask_mode_and_noise_runtime_update_effect():
	ds = _build_ds(mask_ratio=0.5)

	# まず replace: 値が変わる（高確率）
	x1 = ds[None]
	assert len(x1['mask_indices']) == int(0.5 * x1['original'].shape[1])
	# runtime で "add" & noise_std=0.0 → 値は変わらないはず
	ds.mask_mode = 'add'
	ds.mask_noise_std = 0.0
	x2 = ds[None]
	# 形・本数はそのまま
	assert len(x2['mask_indices']) == int(0.5 * x2['original'].shape[1])
	# 追加ノイズが0なので masked == original
	np.testing.assert_allclose(
		x2['masked'].cpu().numpy(),
		x2['original'].cpu().numpy(),
		atol=0.0,
		rtol=0.0,
	)

	ds.close()

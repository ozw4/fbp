import numpy as np

from proc.util.datasets.trace_subset_sampler import (
	TraceSubsetSampler,
	TraceSubsetSamplerConfig,
)


def fake_info():
	# key_to_indices: 2つの ffid グループ、それぞれ 5 本
	ffid_vals = np.array(
		[100, 100, 100, 100, 100, 101, 101, 101, 101, 101], dtype=np.int64
	)
	chno_vals = np.arange(10, dtype=np.int64)
	offsets = np.linspace(0, 9, 10).astype(np.float32)
	key_to_idx = {100: np.arange(0, 5), 101: np.arange(5, 10)}
	uniq = [100, 101]
	return {
		'ffid_values': ffid_vals,
		'chno_values': chno_vals,
		'offsets': offsets,
		'ffid_key_to_indices': key_to_idx,
		'ffid_unique_keys': uniq,
		# セントロイドなしで OK（index window fallback 動作を確認）
		'ffid_centroids': None,
	}


def test_sampler_semantics_without_superwindow():
	info = fake_info()
	cfg = TraceSubsetSamplerConfig(
		primary_keys=('ffid',),
		primary_key_weights=(1.0,),
		use_superwindow=False,
		valid=True,
		subset_traces=8,
	)
	sampler = TraceSubsetSampler(cfg)
	out = sampler.draw(info)  # random は許容（仕様テスト）

	# 仕様：出力項目
	assert set(out.keys()) == {
		'indices',
		'pad_len',
		'key_name',
		'secondary_key',
		'did_super',
		'primary_unique',
	}
	idx = out['indices']
	assert idx.dtype == np.int64
	assert len(idx) <= 8
	assert out['key_name'] == 'ffid'
	assert out['secondary_key'] in {'chno', 'ffid', 'offset'}  # valid=True → "chno"
	assert out['did_super'] in {False, True}  # 今回は False のはず

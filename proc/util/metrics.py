import torch


def prepare_fb_windows(
	fb_idx: torch.Tensor,  # (B,H)  クロップ後のFB。欠損は <=0 推奨
	W: int,  # 時間長
	*,
	pre_len=400,
	post_len=400,
	guard=12,
	device=None,
):
	"""FBから一度だけ「窓インデックス」と「validマスク」を作る。
	返り値を IN/OUT のSNR計算で使い回す。
	"""
	device = device or fb_idx.device
	fb = fb_idx.to(device).long()  # (B,H)

	valid = fb > 0  # 0や負を欠損として除外（必要なら >= に変えてOK）
	valid &= (fb >= guard + pre_len) & (fb + guard + post_len <= W)

	# ノイズ窓 [fb - guard - pre_len, fb - guard)
	n0 = (fb - guard - pre_len).clamp_min(0)
	nseq = torch.arange(pre_len, device=device)
	n_idx = n0[:, :, None] + nseq[None, None, :]  # (B,H,pre_len)

	# 信号窓   [fb + guard, fb + guard + post_len)
	s0 = (fb + guard).clamp_max(W - post_len)
	sseq = torch.arange(post_len, device=device)
	s_idx = s0[:, :, None] + sseq[None, None, :]  # (B,H,post_len)

	return {'n_idx': n_idx, 's_idx': s_idx, 'valid': valid}


@torch.no_grad()
def snr_db_from_cached_windows(
	x: torch.Tensor,  # (B,1,H,W)
	cache: dict,  # prepare_fb_windows の返り値
	reduction: str = 'median',  # or "mean" or None
	eps: float = 1e-12,
):
	"""事前計算した (n_idx, s_idx, valid) を使って SNR を計算。"""
	n_idx, s_idx, valid = cache['n_idx'], cache['s_idx'], cache['valid']
	B, _, H, _ = x.shape
	device = x.device

	# gather のバッチ/トレース添字
	b_idx = torch.arange(B, device=device)[:, None, None]
	h_idx = torch.arange(H, device=device)[None, :, None]

	xin_noise = x[b_idx, :, h_idx, n_idx].squeeze(1)  # (B,H,pre_len)
	xin_sig = x[b_idx, :, h_idx, s_idx].squeeze(1)  # (B,H,post_len)

	pow_sig = (xin_sig**2).mean(dim=-1)
	pow_noi = (xin_noise**2).mean(dim=-1)
	snr_lin = pow_sig / (pow_noi + eps)
	snr_db = 10.0 * torch.log10(snr_lin.clamp_min(eps))  # (B,H)

	if reduction is None:
		return {'snr_db_bh': snr_db, 'valid': valid}

	if valid.any():
		vals = snr_db[valid]
		agg = vals.median() if reduction == 'median' else vals.mean()
	else:
		agg = torch.tensor(float('nan'), device=device)

	return {
		'snr_db_bh': snr_db,
		'snr_db': agg,
		'valid': valid,
		'valid_frac': valid.float().mean(),
	}


@torch.no_grad()
def snr_improvement_from_cached_windows(x_in, x_out, cache, reduction='median'):
	"""ΔSNR を前計算キャッシュで一発算出"""
	r_in = snr_db_from_cached_windows(x_in, cache, reduction=None)
	r_out = snr_db_from_cached_windows(x_out, cache, reduction=None)
	valid = r_in['valid'] & r_out['valid']

	imp_bh = r_out['snr_db_bh'] - r_in['snr_db_bh']  # (B,H)

	if reduction is None:
		return {
			'snr_in_db_bh': r_in['snr_db_bh'],
			'snr_out_db_bh': r_out['snr_db_bh'],
			'snr_improve_db_bh': imp_bh,
			'valid': valid,
		}

	if valid.any():
		snr_in = r_in['snr_db_bh'][valid]
		snr_out = r_out['snr_db_bh'][valid]
		snr_imp = imp_bh[valid]
		agg_in = snr_in.median() if reduction == 'median' else snr_in.mean()
		agg_out = snr_out.median() if reduction == 'median' else snr_out.mean()
		agg_imp = snr_imp.median() if reduction == 'median' else snr_imp.mean()
	else:
		nan = torch.tensor(float('nan'), device=x_in.device)
		agg_in = agg_out = agg_imp = nan

	return {
		'snr_in_db': agg_in,
		'snr_out_db': agg_out,
		'snr_improve_db': agg_imp,
		'valid_frac': valid.float().mean(),
	}

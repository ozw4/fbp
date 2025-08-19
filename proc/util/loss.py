# %%
# Fix: expand the leading K dimension before gather
import time

import torch


def criterion(
	pred,
	gt,
	mask=None,  # (B,1,H,W) 推奨（MAEのマスクを渡す）
	max_shift=5,
	reduction='mean',
):
	"""主損失: トレース毎のシフト探索つきL2（MSE）
	- mask を渡すとその領域のみで計算
	- aux_outputs があれば同じロスを平均して 0.4 倍で加算（重みは引数で変更可）
	返り値: (total_loss, loss_main, loss_l2_plain)  ← 3つ目はログ用の通常MSE
	"""
	# 主損失（ロバストL2）
	loss = shift_robust_l2_pertrace_vec(
		pred, gt, mask=mask, max_shift=max_shift, reduction=reduction
	)
	# loss_l1_plain = l1loss(pred, gt).detach() if 'l1loss' in globals() else torch.tensor(0.0, device=pred.device)

	return loss


def shift_robust_l2_pertrace_vec(
	pred: torch.Tensor,  # (B, C, H, W)
	gt: torch.Tensor,  # (B, C, H, W)
	mask: torch.Tensor | None = None,  # (B,1,H,W) or (B,C,H,W) or None
	max_shift: int = 6,
	reduction: str = 'mean',  # 'mean' | 'sum' | 'none'
):
	assert pred.shape == gt.shape, 'pred and gt must have the same shape'
	B, C, H, W = pred.shape
	device, dtype = pred.device, pred.dtype

	S = int(max_shift)
	if S < 0:
		raise ValueError('max_shift must be >= 0')
	if W - S <= 0:
		raise ValueError(
			f'max_shift={S} is too large for width W={W} (need W > max_shift).'
		)

	K = 2 * S + 1  # number of shifts
	minW = W - S  # common length across shifts

	s_offsets = torch.arange(-S, S + 1, device=device)  # (K,)
	start_pred = torch.clamp(s_offsets, min=0)  # (K,)
	start_gt = torch.clamp(-s_offsets, min=0)  # (K,)

	base_idx = torch.arange(minW, device=device)  # (minW,)
	idx_pred = start_pred[:, None] + base_idx[None, :]  # (K, minW)
	idx_gt = start_gt[:, None] + base_idx[None, :]  # (K, minW)

	idxp = idx_pred.view(K, 1, 1, 1, minW).expand(-1, B, C, H, -1)  # (K,B,C,H,minW)
	idxg = idx_gt.view(K, 1, 1, 1, minW).expand(-1, B, C, H, -1)

	pred_expanded = pred.unsqueeze(0).expand(K, -1, -1, -1, -1)  # (K,B,C,H,W)
	gt_expanded = gt.unsqueeze(0).expand(K, -1, -1, -1, -1)  # (K,B,C,H,W)

	pred_g = pred_expanded.gather(dim=-1, index=idxp)  # (K,B,C,H,minW)
	gt_g = gt_expanded.gather(dim=-1, index=idxg)  # (K,B,C,H,minW)

	if mask is not None:
		if mask.dim() != 4:
			mask = mask.view(B, 1, H, W)
		mask = mask.to(dtype=dtype)
		if mask.size(1) == 1:
			mask_expanded = mask.unsqueeze(0).expand(K, -1, -1, -1, -1)  # (K,B,1,H,W)
			idxm = idxg[:, :, :1, :, :]
		elif mask.size(1) == C:
			mask_expanded = mask.unsqueeze(0).expand(K, -1, -1, -1, -1)  # (K,B,C,H,W)
			idxm = idxg
		else:
			raise ValueError('mask channel dimension must be 1 or match C.')
		mask_g = mask_expanded.gather(dim=-1, index=idxm)  # (K,B,1orC,H,minW)
	else:
		mask_g = None

	diff2 = (pred_g - gt_g) ** 2  # (K,B,C,H,minW)
	if mask_g is not None:
		num = (diff2 * mask_g).sum(dim=(2, 4))  # (K,B,H)
		den = mask_g.sum(dim=(2, 4)).clamp_min(1e-8)  # (K,B,H)
		loss_kbh = num / den  # (K,B,H)
	else:
		loss_kbh = diff2.mean(dim=(2, 4))  # (K,B,H)

	best_bh = loss_kbh.min(dim=0).values  # (B,H)

	if reduction == 'none':
		return best_bh
	if reduction == 'sum':
		return best_bh.sum()
	return best_bh.mean()


# Reuse the earlier loop reference and tests
def shift_robust_l2_pertrace_loop(pred, gt, mask=None, max_shift=6, reduction='mean'):
	assert pred.shape == gt.shape
	B, C, H, W = pred.shape
	device, dtype = pred.device, pred.dtype
	if mask is not None:
		if mask.dim() != 4:
			mask = mask.view(B, 1, H, W)
		mask = mask.to(dtype=dtype)

	best = torch.full((B, H), float('inf'), device=device, dtype=dtype)
	for s in range(-max_shift, max_shift + 1):
		if s == 0:
			ps, gs = slice(None), slice(None)
		elif s > 0:
			if s >= W:
				continue
			ps, gs = slice(s, None), slice(None, -s)
		else:
			if -s >= W:
				continue
			ps, gs = slice(None, s), slice(-s, None)

		pd, gd = pred[..., ps], gt[..., gs]
		if mask is not None:
			md = mask[..., gs] if s >= 0 else mask[..., ps]
			diff2 = (pd - gd) ** 2  # (B,C,H,W')
			num = (diff2 * md).sum(dim=(1, 3))  # (B,H)
			den = md.sum(dim=(1, 3)).clamp_min(1e-8)  # (B,H)
			loss_bh = num / den  # (B,H)
		else:
			loss_bh = ((pd - gd) ** 2).mean(dim=(1, 3))  # (B,H)

		best = torch.minimum(best, loss_bh)

	if reduction == 'none':
		return best
	if reduction == 'sum':
		return best.sum()
	return best.mean()


def run_tests():
	torch.manual_seed(42)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	B, C, H, W = 3, 1, 8, 64
	S = 6

	base = torch.randn(B, C, H, W, device=device)

	# 1) Identity
	pred = base.clone()
	gt = base.clone()
	loss_vec = shift_robust_l2_pertrace_vec(pred, gt, None, S, 'mean')
	loss_ref = shift_robust_l2_pertrace_loop(pred, gt, None, S, 'mean')
	print('[Identity] vec:', float(loss_vec), 'ref:', float(loss_ref))

	# 2) Known shift
	shift = 3
	pred2 = torch.zeros_like(gt)
	pred2[..., shift:] = gt[..., :-shift]
	pred2 += 0.001 * torch.randn_like(pred2)
	loss_vec2 = shift_robust_l2_pertrace_vec(pred2, gt, None, S, 'mean')
	loss_ref2 = shift_robust_l2_pertrace_loop(pred2, gt, None, S, 'mean')
	print('[Known shift] vec:', float(loss_vec2), 'ref:', float(loss_ref2))

	# 3) Masked right half
	mask = torch.zeros(B, 1, H, W, device=device)
	mask[..., W // 2 :] = 1.0
	loss_vec3 = shift_robust_l2_pertrace_vec(pred2, gt, mask, S, 'mean')
	loss_ref3 = shift_robust_l2_pertrace_loop(pred2, gt, mask, S, 'mean')
	print('[Masked right-half] vec:', float(loss_vec3), 'ref:', float(loss_ref3))

	# 4) Reduction='none'
	loss_vec4 = shift_robust_l2_pertrace_vec(pred2, gt, None, S, 'none')
	loss_ref4 = shift_robust_l2_pertrace_loop(pred2, gt, None, S, 'none')
	print(
		'[Reduction none] shapes:',
		loss_vec4.shape,
		loss_ref4.shape,
		' max|diff|=',
		float((loss_vec4 - loss_ref4).abs().max()),
	)

	# 5) C>1 with C-channel mask
	C2 = 3
	base2 = torch.randn(B, C2, H, W, device=device)
	predC = base2.clone()
	gtC = base2.clone()
	maskC = torch.ones(B, C2, H, W, device=device)
	maskC[:, 0] = 0.0  # ignore channel 0
	loss_vec5 = shift_robust_l2_pertrace_vec(predC, gtC, maskC, S, 'mean')
	loss_ref5 = shift_robust_l2_pertrace_loop(predC, gtC, maskC, S, 'mean')
	print('[Multi-channel mask] vec:', float(loss_vec5), 'ref:', float(loss_ref5))

	# 6) Timing comparison
	Bt, Ct, Ht, Wt, St = 4, 1, 128, 512, 6
	A = torch.randn(Bt, Ct, Ht, Wt, device=device)
	Bsig = torch.randn(Bt, Ct, Ht, Wt, device=device)
	start = time.time()
	for _ in range(10):
		_ = shift_robust_l2_pertrace_vec(A, Bsig, None, St, 'mean')
	t_vec = time.time() - start

	start = time.time()
	for _ in range(10):
		_ = shift_robust_l2_pertrace_loop(A, Bsig, None, St, 'mean')
	t_loop = time.time() - start
	print(
		f'[Timing 10 iters] vec={t_vec:.4f}s  loop={t_loop:.4f}s  speedup×{t_loop / max(t_vec, 1e-9):.2f}'
	)


if __name__ == '__main__':
	run_tests()

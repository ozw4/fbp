# %%
# Fix: expand the leading K dimension before gather
import time

import torch
import torch.nn.functional as F

from proc.util.velocity_mask import make_velocity_feasible_mask


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


def compute_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	mask: torch.Tensor | None = None,
	cfg_loss=None,
):
	"""Compute training loss.

	Uses standard MSE by default, and switches to shift-robust MSE when
	``cfg_loss.shift_robust`` is True. The existing implementation
	``shift_robust_l2_pertrace_vec`` is reused for the robust variant.
	"""
	shift_robust = bool(getattr(cfg_loss, 'shift_robust', False))
	max_shift = int(getattr(cfg_loss, 'max_shift', 5))
	if shift_robust:
		return shift_robust_l2_pertrace_vec(
			pred, target, mask=mask, max_shift=max_shift, reduction='mean'
		)
	# standard masked/unmasked MSE
	if mask is not None:
		if mask.dim() != 4:
			mask = mask.view(pred.size(0), 1, pred.size(2), pred.size(3))
		if mask.size(1) == 1:
			mask = mask.expand(-1, pred.size(1), -1, -1)
		diff2 = (pred - target) ** 2
		masked = diff2 * mask
		return masked.sum() / mask.sum().clamp_min(1e-8)
	return F.mse_loss(pred, target)


def make_criterion(cfg_loss):
	"""Return a criterion compatible with ``train_one_epoch``."""

	def _criterion(pred, target, *, mask=None, **kwargs):
		return compute_loss(pred, target, mask=mask, cfg_loss=cfg_loss)

	return _criterion


def fb_seg_kl_loss(
	logits: torch.Tensor,
	target: torch.Tensor,
	valid_mask: torch.Tensor | None = None,
	tau: float = 1.0,
	eps: float = 0.0,
):
	"""KL-divergence loss for first-break segmentation.

	Parameters
	----------
	logits : torch.Tensor
			Raw model outputs of shape ``(B,1,H,W)``.
	target : torch.Tensor
			Target probabilities (Gaussian-like) of shape ``(B,1,H,W)``.
	valid_mask : torch.Tensor | None
			Optional mask indicating valid traces. Expected shape ``(B,H)``
			or ``(B,1,H,1)``. Invalid traces are ignored in the loss.
	tau : float
			Softmax temperature.
	eps : float
			Label smoothing factor.

	"""
	log_p = F.log_softmax(logits.squeeze(1) / tau, dim=-1)  # (B,H,W)
	q = target.squeeze(1)
	q = q / (q.sum(dim=-1, keepdim=True) + 1e-12)
	if eps > 0:
		q = (1 - eps) * q + eps / q.size(-1)
	kl_bh = -(q * log_p).sum(dim=-1)  # (B,H)
	if valid_mask is not None:
		if valid_mask.dim() == 4:
			valid = valid_mask.squeeze(1).squeeze(-1)
		else:
			valid = valid_mask
		valid = valid.to(kl_bh.dtype)
		num = (kl_bh * valid).sum()
		den = valid.sum().clamp_min(1)
		return num / den
	return kl_bh.mean()


def make_fb_seg_criterion(cfg_fb):
	"""Factory for fb segmentation loss.

	``cfg_fb.type`` selects between KL-divergence (``'kl'``) and MSE
	(``'mse'``). Additional parameters ``tau`` and ``eps`` configure the
	KL variant.
	"""
	fb_type = str(getattr(cfg_fb, 'type', 'kl')).lower()
	if fb_type == 'kl':
		tau = float(getattr(cfg_fb, 'tau', 1.0))
		eps = float(getattr(cfg_fb, 'eps', 0.0))
		smooth_lambda = float(getattr(cfg_fb, 'smooth_lambda', 0.0))
		smooth_weight = str(getattr(cfg_fb, 'smooth_weight', 'inv'))
		smooth_scale = float(getattr(cfg_fb, 'smooth_scale', 1.0))
		smooth2_lambda = float(getattr(cfg_fb, 'smooth2_lambda', 0.0))

		def _criterion(
			pred: torch.Tensor,
			target: torch.Tensor,
			*,
			fb_idx: torch.Tensor,
			mask=None,
			offsets=None,
			dt_sec: torch.Tensor | None = None,
			**kwargs,
		) -> torch.Tensor | tuple[torch.Tensor, dict]:
			valid_mask = (fb_idx >= 0).to(pred.dtype)

			# --- build masked logits by applying a velocity cone prior ---
			logit_raw = pred.squeeze(1)  # (B,H,W)
			B, H, W = logit_raw.shape

			use_vel_mask = bool(getattr(cfg_fb, 'use_vel_mask', False))
			if use_vel_mask and (offsets is not None) and (dt_sec is not None):
				velmask = make_velocity_feasible_mask(
					offsets=offsets,
					dt_sec=dt_sec,
					W=W,
					vmin=float(getattr(cfg_fb, 'vmin_mask', 500.0)),
					vmax=float(getattr(cfg_fb, 'vmax_mask', 6000.0)),
					t0_lo_ms=float(getattr(cfg_fb, 't0_lo_ms', -20.0)),
					t0_hi_ms=float(getattr(cfg_fb, 't0_hi_ms', 80.0)),
					taper_ms=float(getattr(cfg_fb, 'taper_ms', 10.0)),
					device=pred.device,
					dtype=pred.dtype,
				)

				# ---- SAFE log(mask): fp32で計算 + fp16安全なepsilon + 全ゼロ対策 ----
				with torch.no_grad():
					vm32 = velmask.detach().to(torch.float32)  # (B,H,W)
					# 全ゼロ・トレースを検出
					has_any = vm32.sum(dim=-1, keepdim=True) > 0  # (B,H,1) bool
					# 全ゼロのトレースは「マスク無効（=全1）」に差し替え
					vm32 = torch.where(has_any, vm32, torch.ones_like(vm32))

					eps = float(getattr(cfg_fb, 'vel_log_eps', 1e-4))  # ★ fp16安全
					neg_large = float(
						getattr(cfg_fb, 'vel_log_neg', -80.0)
					)  # ★ ゼロには一定の大負値
					logmask32 = torch.where(
						vm32 > 0.0,
						torch.log(vm32.clamp_min(eps)),
						torch.full_like(vm32, neg_large),
					)

				# logits + log(mask)（元dtypeへ）
				logit = logit_raw + logmask32.to(logit_raw.dtype)
			else:
				logit = logit_raw

			# --- base KL (use masked logits) ---
			base = fb_seg_kl_loss(
				logit.unsqueeze(1), target, valid_mask=valid_mask, tau=tau, eps=eps
			)

			# downstream uses the masked probabilities
			prob = torch.softmax(logit / tau, dim=-1)  # (B,H,W)

			if (smooth_lambda <= 0 and smooth2_lambda <= 0) or offsets is None:
				return base, {
					'base': base.detach(),
					'smooth': base.new_tensor(0.0),
					'curv': base.new_tensor(0.0),
				}

			W = prob.size(-1)
			t = torch.arange(W, device=prob.device, dtype=prob.dtype)
			pos = (prob * t).sum(dim=-1)  # (B,H)
			if dt_sec is None:
				dt = pos.new_ones((pos.size(0), 1))
			else:
				dt = dt_sec.to(pos.device, pos.dtype)
				dt = dt.view(dt.size(0), 1)
			pos_sec = pos * dt  # (B,H)

			dx = (offsets[:, 1:] - offsets[:, :-1]).abs().to(pos_sec.dtype)
			# データ適応の安全下限：メディアンの1e-4倍 or 1mm の大きい方
			dx_med = dx.median()
			dx_eps = torch.clamp(
				dx_med * 1e-4, min=torch.tensor(1e-3, device=dx.device, dtype=dx.dtype)
			)
			dx_safe = dx.clamp_min(dx_eps)

			smooth = base.new_tensor(0.0)
			if smooth_lambda > 0:
				dpos = pos_sec[:, 1:] - pos_sec[:, :-1]
				scale = dx.median(dim=1, keepdim=True).values.clamp_min(1.0)
				if smooth_weight == 'inv':
					w = 1.0 / (dx / scale + smooth_scale)
				else:
					w = torch.exp(-(dx / scale) / max(smooth_scale, 1e-6))
				v2 = (valid_mask[:, 1:] * valid_mask[:, :-1]).to(dpos.dtype)
				num = (w * v2 * (dpos**2)).sum()
				den = (w * v2).sum().clamp_min(1.0)
				smooth = num / den

			loss_curv = base.new_tensor(0.0)
			if smooth2_lambda > 0:
				dpos = pos_sec[:, 1:] - pos_sec[:, :-1]
				s = dpos / dx_safe
				dx_mid = 0.5 * (dx_safe[:, 1:] + dx_safe[:, :-1]).clamp_min(dx_eps)
				curv = (s[:, 1:] - s[:, :-1]) / dx_mid
				v3 = (valid_mask[:, 2:] * valid_mask[:, 1:-1] * valid_mask[:, :-2]).to(
					curv.dtype
				)
				dx_ok = (dx > dx_eps).to(curv.dtype)
				v3 = v3 * dx_ok[:, 1:] * dx_ok[:, :-1]
				scale2 = dx_mid.median(dim=1, keepdim=True).values.clamp_min(1.0)
				if smooth_weight == 'inv':
					w2 = 1.0 / ((dx_mid / scale2) + smooth_scale)
				else:
					w2 = torch.exp(-(dx_mid / scale2) / max(smooth_scale, 1e-6))
				num2 = (w2 * v3 * (curv**2)).sum()
				den2 = (w2 * v3).sum().clamp_min(1.0)
				loss_curv = num2 / den2

			total = base + smooth_lambda * smooth + smooth2_lambda * loss_curv
			return total, {
				'base': base.detach(),
				'smooth': (smooth_lambda * smooth).detach(),
				'curv': (smooth2_lambda * loss_curv).detach(),
			}

		return _criterion

	if fb_type == 'mse':
		return make_fb_seg_mse_criterion()

	raise ValueError(f'Unknown fb_seg loss type: {fb_type}')


def make_fb_seg_mse_criterion():
	"""Return MSE criterion for fb segmentation.

	The loss averages the MSE over traces where ``fb_idx>=0``. If no valid
	trace exists in the batch, the loss gracefully returns ``0`` without
	raising errors or producing NaNs.
	"""

	def _criterion(
		pred: torch.Tensor,
		target: torch.Tensor,
		*,
		fb_idx: torch.Tensor,
		mask=None,
		**kwargs,
	) -> torch.Tensor:
		# pred/target: (B,1,H,W), fb_idx: (B,H)
		diff2 = (pred - target) ** 2  # (B,1,H,W)
		loss_bh = diff2.mean(dim=(1, 3))  # (B,H)
		valid = (fb_idx >= 0).to(loss_bh.dtype)
		num = (loss_bh * valid).sum()
		den = valid.sum().clamp_min(1)
		return num / den

	return _criterion


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

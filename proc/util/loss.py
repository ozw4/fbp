# %%

# %%

import math
import time

import torch
import torch.nn.functional as F

from proc.util.velocity_mask import make_velocity_feasible_mask


@torch.no_grad()
def trace_confidence_from_prob(
	prob: torch.Tensor,  # (B,H,W) after softmax
	dt_sec: torch.Tensor,  # (B,1) or (B,)
	method: str = 'var',  # "var" | "entropy" | "var+ent"
	sigma_ms_ref: float = 20.0,  # variance reference (ms)
	floor: float = 0.2,  # min weight to avoid self-reinforcement
	power: float = 0.5,  # soften (0.5 = sqrt)
	eps: float = 1e-9,
) -> torch.Tensor:
	"""Build per-trace confidence weights (B,H) from probability p(t); no grad."""
	B, H, W = prob.shape
	t_idx = torch.arange(W, device=prob.device, dtype=prob.dtype).view(1, 1, W)
	dt = dt_sec.to(prob).view(B, 1, 1)
	t = t_idx * dt  # (B,1,W) [s]

	mu = (prob * t).sum(dim=-1)  # (B,H)
	var = (prob * (t - mu.unsqueeze(-1)) ** 2).sum(dim=-1)  # (B,H)
	var = var.clamp_min(eps)

	w_var = None
	if method in ('var', 'var+ent'):
		sigma_ref = max(sigma_ms_ref * 1e-3, 1e-6)
		w_var = torch.exp(-var / (2 * sigma_ref**2))

	if method == 'var':
		w = w_var
	elif method == 'entropy':
		Hlog = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=-1)
		Hnorm = Hlog / math.log(W)
		w = (1.0 - Hnorm).clamp(0.0, 1.0)
	else:  # "var+ent"
		Hlog = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=-1)
		Hnorm = Hlog / math.log(W)
		w_ent = (1.0 - Hnorm).clamp(0.0, 1.0)
		w = (w_var * w_ent).sqrt()

	return w.clamp_min(floor) ** power


@torch.no_grad()
def robust_linear_trend_sections(
	offsets: torch.Tensor,  # (B,H) [m]
	t_sec: torch.Tensor,  # (B,H) predicted pos_sec [s]
	valid: torch.Tensor,  # (B,H) fb_idx>=0
	*,
	prob: torch.Tensor,  # (B,H,W) after softmax
	dt_sec: torch.Tensor,  # (B,1) or (B,)
	section_len: int = 128,
	stride: int = 64,
	huber_c: float = 1.345,
	iters: int = 3,
	vmin: float = 300.0,
	vmax: float = 6000.0,
	# confidence construction
	conf_method: str = 'var',  # "var" | "entropy" | "var+ent"
	conf_sigma_ms: float = 20.0,
	conf_floor: float = 0.2,
	conf_power: float = 0.5,
	# sorting & blending
	sort_offsets: bool = True,
	use_taper: bool = True,
):
	"""Windowed IRLS (weighted by confidence) to fit t(x) ≈ a + s x.
	Returns: (trend_t, trend_s, v_trend, w_conf, covered) all (B,H)
	"""
	B, H = offsets.shape
	x0 = offsets
	y0 = t_sec
	v0 = (valid > 0).to(t_sec)

	# 1) per-trace confidence (B,H)
	w_conf = (
		trace_confidence_from_prob(
			prob=prob,
			dt_sec=dt_sec,
			method=conf_method,
			sigma_ms_ref=conf_sigma_ms,
			floor=conf_floor,
			power=conf_power,
		)
		.detach()
		.to(t_sec)
	)

	# 2) optional sort by offsets (internal only)
	if sort_offsets:
		idx = torch.argsort(x0, dim=1)
		arangeH = torch.arange(H, device=idx.device).unsqueeze(0).expand_as(idx)
		inv = torch.empty_like(idx)
		inv.scatter_(1, idx, arangeH)

		x = torch.gather(x0, 1, idx)
		y = torch.gather(y0, 1, idx)
		v = torch.gather(v0, 1, idx)
		pw = torch.gather(w_conf, 1, idx)
	else:
		x, y, v, pw = x0, y0, v0, w_conf

	trend_t = torch.zeros_like(y)
	trend_s = torch.zeros_like(y)
	counts = torch.zeros_like(y)

	eps = 1e-12
	for start in range(0, H, stride):
		end = min(H, start + section_len)
		L = end - start
		if L < 4:
			continue

		xs = x[:, start:end]  # (B,L)
		ys = y[:, start:end]
		vs = v[:, start:end]
		pws = pw[:, start:end]  # (B,L)

		# init weights: valid × confidence
		w = (vs * pws).clone()

		a = torch.zeros(B, 1, dtype=y.dtype, device=y.device)
		b = torch.zeros(B, 1, dtype=y.dtype, device=y.device)  # slope=slowness

		for _ in range(iters):
			Sw = (w).sum(dim=1, keepdim=True).clamp_min(eps)
			Sx = (w * xs).sum(dim=1, keepdim=True)
			Sy = (w * ys).sum(dim=1, keepdim=True)
			Sxx = (w * xs * xs).sum(dim=1, keepdim=True)
			Sxy = (w * xs * ys).sum(dim=1, keepdim=True)

			D = (Sw * Sxx - Sx * Sx).clamp_min(eps)
			b = (Sw * Sxy - Sx * Sy) / D
			a = (Sy - b * Sx) / Sw

			yhat = a + b * xs
			res = (ys - yhat) * vs
			# robust scale (MAD)
			scale = (1.4826 * res.abs().median(dim=1, keepdim=True).values).clamp_min(
				1e-6
			)
			r = res / (huber_c * scale)

			w_huber = torch.where(
				r.abs() <= 1.0, vs, vs * (1.0 / r.abs()).clamp_max(10.0)
			)
			# keep confidence as front weights (stop-grad)
			w = w_huber * pws

		# clamp to physical range
		s_sec = b.squeeze(1).clamp(min=1.0 / vmax, max=1.0 / vmin)  # (B,)

		# Hann taper for window blending
		if use_taper:
			wwin = torch.hann_window(
				L, periodic=False, device=y.device, dtype=y.dtype
			).view(1, L)
		else:
			wwin = torch.ones(1, L, device=y.device, dtype=y.dtype)
		wtap = wwin * vs * pws

		yhat = a + b * xs  # (B,L)
		trend_t[:, start:end] += yhat * wtap
		trend_s[:, start:end] += s_sec[:, None] * wtap
		counts[:, start:end] += wtap

	trend_t = trend_t / counts.clamp_min(1e-6)
	trend_s = trend_s / counts.clamp_min(1e-6)
	v_trend = 1.0 / trend_s.clamp_min(1e-6)
	covered = counts > 0

	if sort_offsets:
		trend_t = torch.gather(trend_t, 1, inv)
		trend_s = torch.gather(trend_s, 1, inv)
		v_trend = torch.gather(v_trend, 1, inv)
		w_conf = torch.gather(w_conf, 1, inv)
		covered = torch.gather(covered.to(torch.bool), 1, inv)

	return trend_t, trend_s, v_trend, w_conf, covered


def gaussian_prior_from_trend(
	t_trend_sec: torch.Tensor,  # (B,H)
	dt_sec: torch.Tensor,  # (B,1) or (B,)
	W: int,
	sigma_ms: float,
	ref_tensor: torch.Tensor,
	covered_mask: torch.Tensor | None = None,  # (B,H)
):
	"""Make per-trace Gaussian prior in time around t_trend. Returns (B,H,W) and sums to 1 per trace."""
	B, H = t_trend_sec.shape
	t = torch.arange(W, device=ref_tensor.device, dtype=ref_tensor.dtype).view(1, 1, W)
	if dt_sec.dim() == 1:
		dt = dt_sec.view(B, 1, 1).to(ref_tensor)
	else:
		dt = dt_sec.to(ref_tensor).view(B, 1, 1)

	mu = t_trend_sec.to(ref_tensor).unsqueeze(-1)  # (B,H,1)
	sigma = max(sigma_ms * 1e-3, 1e-6)
	logp = -0.5 * ((t * dt - mu) / sigma) ** 2  # (B,H,W)
	prior = torch.softmax(logp, dim=-1)  # (B,H,W)

	if covered_mask is not None:
		uni = prior.new_full((1, 1, W), 1.0 / W)
		prior = torch.where(covered_mask.to(torch.bool).unsqueeze(-1), prior, uni)

	return prior


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
		raise ValueError(f'max_shift={S} is too large for W={W} (need W > max_shift).')

	K = 2 * S + 1
	minW = W - S

	s_offsets = torch.arange(-S, S + 1, device=device)  # (K,)
	start_pred = torch.clamp(s_offsets, min=0)  # (K,)
	start_gt = torch.clamp(-s_offsets, min=0)  # (K,)

	base_idx = torch.arange(minW, device=device)  # (minW,)
	idx_pred = start_pred[:, None] + base_idx[None, :]  # (K, minW)
	idx_gt = start_gt[:, None] + base_idx[None, :]  # (K, minW)

	idxp = idx_pred.view(K, 1, 1, 1, minW).expand(-1, B, C, H, -1)
	idxg = idx_gt.view(K, 1, 1, 1, minW).expand(-1, B, C, H, -1)

	pred_expanded = pred.unsqueeze(0).expand(K, -1, -1, -1, -1)
	gt_expanded = gt.unsqueeze(0).expand(K, -1, -1, -1, -1)

	pred_g = pred_expanded.gather(dim=-1, index=idxp)  # (K,B,C,H,minW)
	gt_g = gt_expanded.gather(dim=-1, index=idxg)  # (K,B,C,H,minW)

	if mask is not None:
		if mask.dim() != 4:
			mask = mask.view(B, 1, H, W)
		mask = mask.to(dtype=dtype)
		if mask.size(1) == 1:
			mask_expanded = mask.unsqueeze(0).expand(K, -1, -1, -1, -1)
			idxm = idxg[:, :, :1, :, :]
		elif mask.size(1) == C:
			mask_expanded = mask.unsqueeze(0).expand(K, -1, -1, -1, -1)
			idxm = idxg
		else:
			raise ValueError('mask channel dimension must be 1 or match C.')
		mask_g = mask_expanded.gather(dim=-1, index=idxm)
	else:
		mask_g = None

	diff2 = (pred_g - gt_g) ** 2
	if mask_g is not None:
		num = (diff2 * mask_g).sum(dim=(2, 4))  # (K,B,H)
		den = mask_g.sum(dim=(2, 4)).clamp_min(1e-8)  # (K,B,H)
		loss_kbh = num / den
	else:
		loss_kbh = diff2.mean(dim=(2, 4))

	best_bh = loss_kbh.min(dim=0).values  # (B,H)

	if reduction == 'none':
		return best_bh
	if reduction == 'sum':
		return best_bh.sum()
	return best_bh.mean()


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
			diff2 = (pd - gd) ** 2
			num = (diff2 * md).sum(dim=(1, 3))  # (B,H)
			den = md.sum(dim=(1, 3)).clamp_min(1e-8)
			loss_bh = num / den
		else:
			loss_bh = ((pd - gd) ** 2).mean(dim=(1, 3))

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
	"""Standard MSE or shift-robust MSE depending on cfg."""
	shift_robust = bool(getattr(cfg_loss, 'shift_robust', False))
	max_shift = int(getattr(cfg_loss, 'max_shift', 5))
	if shift_robust:
		p32 = pred.to(torch.float32)
		t32 = target.to(torch.float32)
		m32 = None
		if mask is not None:
			if mask.dim() != 4:
				mask = mask.view(pred.size(0), 1, pred.size(2), pred.size(3))
			m32 = mask.to(torch.float32)

		return shift_robust_l2_pertrace_vec(
			p32, t32, mask=m32, max_shift=max_shift, reduction='mean'
		)
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
	"""Return a criterion compatible with train_one_epoch."""

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
	"""KL-divergence loss for first-break segmentation."""
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
	"""Factory for fb segmentation loss with safe velocity mask & trend prior."""
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
			logit_clip = float(getattr(cfg_fb, 'logit_clip', 30.0))

			# --- base logits (B,H,W) ------------------------------------------
			logit_raw = pred.squeeze(1)  # (B,H,W)
			B, H, W = logit_raw.shape

			# --- optional velocity-feasible mask as log-add -------------------
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

				# SAFE log(mask) in fp32 with strict sanitization
				with torch.no_grad():
					vm32 = velmask.detach().to(torch.float32)  # (B,H,W)
					vm32 = torch.nan_to_num(
						vm32, nan=0.0, posinf=0.0, neginf=0.0
					).clamp_(0.0, 1.0)
					has_any = vm32.sum(dim=-1, keepdim=True) > 0  # (B,H,1)
					vm32 = torch.where(has_any, vm32, torch.ones_like(vm32))

					vel_log_eps = float(getattr(cfg_fb, 'vel_log_eps', 1e-4))
					neg_large = float(getattr(cfg_fb, 'vel_log_neg', -80.0))
					logmask32 = torch.where(
						vm32 > 0.0,
						torch.log(vm32.clamp_min(vel_log_eps)),
						torch.full_like(vm32, neg_large),
					)

				# add in fp32, then sanitize & clip
				logit = (logit_raw.to(torch.float32) + logmask32).to(logit_raw.dtype)
			else:
				logit = logit_raw

			# sanitize & clip logits BEFORE any prior
			logit = torch.nan_to_num(
				logit, nan=0.0, posinf=logit_clip, neginf=-logit_clip
			).clamp_(-logit_clip, logit_clip)

			# --- optional trend-based prior ----------------------------------
			prior_mode = str(getattr(cfg_fb, 'prior_mode', 'logit')).lower()
			prior_alpha = float(getattr(cfg_fb, 'prior_alpha', 0.1))
			prior_ce = logit.new_tensor(0.0)
			covered = torch.zeros(B, H, dtype=torch.bool, device=logit.device)

			use_trend_prior = bool(getattr(cfg_fb, 'use_trend_prior', False))
			have_prior = (
				use_trend_prior
				and (offsets is not None)
				and (dt_sec is not None)
				and (prior_alpha > 0)
			)

			log_prior = None
			prior = None

			if have_prior:
				prob_for_trend = torch.softmax(logit / tau, dim=-1)
				if not torch.isfinite(prob_for_trend).all():
					raise FloatingPointError('[NaNGuard] prob_for_trend non-finite')

				t_idx = torch.arange(W, device=logit.device, dtype=prob_for_trend.dtype)
				pos = (prob_for_trend * t_idx).sum(dim=-1)  # (B,H)
				dt = dt_sec.to(pos.device, pos.dtype).view(pos.size(0), 1)
				pos_sec = pos * dt

				with torch.no_grad():
					t_tr, s_tr, v_tr, w_conf, covered = robust_linear_trend_sections(
						offsets=offsets.to(pos_sec),
						t_sec=pos_sec,
						valid=(fb_idx >= 0),
						section_len=int(getattr(cfg_fb, 'trend_section', 128)),
						stride=int(getattr(cfg_fb, 'trend_stride', 64)),
						huber_c=float(getattr(cfg_fb, 'trend_huber_c', 1.345)),
						iters=3,
						vmin=float(getattr(cfg_fb, 'trend_vmin', 500.0)),
						vmax=float(getattr(cfg_fb, 'trend_vmax', 5000.0)),
						prob=prob_for_trend,
						dt_sec=dt_sec,
					)

				prior = gaussian_prior_from_trend(
					t_trend_sec=t_tr,
					dt_sec=dt_sec,
					W=W,
					sigma_ms=float(getattr(cfg_fb, 'prior_sigma_ms', 20.0)),
					ref_tensor=logit,
					covered_mask=covered,
				)

				# normalize & safe log in fp32
				prior = torch.nan_to_num(prior, nan=0.0).clamp_(min=0.0)
				prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-12)

				prior_log_eps = float(getattr(cfg_fb, 'prior_log_eps', 1e-4))
				log_prior = torch.log(prior.clamp_min(prior_log_eps)).to(torch.float32)

				# simple confidence gate (median over valid traces)
				epsH = 1e-9
				Hent = -(
					prob_for_trend.clamp_min(epsH)
					* prob_for_trend.clamp_min(epsH).log()
				).sum(dim=-1)
				Hnorm = Hent / math.log(prob_for_trend.size(-1))
				conf = 1.0 - Hnorm
				conf_med = (
					conf[(fb_idx >= 0)].median()
					if (fb_idx >= 0).any()
					else conf.median()
				)
				gate_th = float(getattr(cfg_fb, 'prior_conf_gate', 0.5))
				if conf_med < gate_th:
					prior_alpha = 0.0  # gate off prior if low confidence

			# --- combine prior ------------------------------------------------
			if prior_mode == 'logit':
				# 保存: prior合成前（既にsanitize/clip済み）の安全なlogit
				logit_base = logit.clone()

				if have_prior and prior_alpha > 0.0 and log_prior is not None:
					# まず prior 側も有限かチェック
					if not torch.isfinite(log_prior).all():
						# prior が壊れている → このバッチは prior 無効化
						if bool(getattr(cfg_fb, 'prior_debug', False)):
							print(
								'[trend prior] non-finite log_prior -> disable prior for this batch'
							)
						prior_alpha = 0.0
						logit = logit_base
					else:
						# FP32で合成してからclip→dtype復帰
						device_type = 'cuda' if logit.is_cuda else 'cpu'
						with torch.autocast(device_type=device_type, enabled=False):
							logit32 = (
								logit.to(torch.float32) + prior_alpha * log_prior
							)  # fp32
							logit32 = torch.nan_to_num(
								logit32, nan=0.0, posinf=logit_clip, neginf=-logit_clip
							).clamp_(-logit_clip, logit_clip)

						if not torch.isfinite(logit32).all():
							# ★ フォールバック: priorを無効化して続行
							if bool(getattr(cfg_fb, 'prior_debug', False)):
								bad = (~torch.isfinite(logit32)).sum().item()
								print(
									f'[trend prior] non-finite after add (count={bad}) -> disable prior for this batch'
								)
							prior_alpha = 0.0
							logit = logit_base
						else:
							logit = logit32.to(logit.dtype)
				# 最終ガード（prior無効でも一応）
				logit = torch.nan_to_num(
					logit, nan=0.0, posinf=logit_clip, neginf=-logit_clip
				).clamp_(-logit_clip, logit_clip)

			elif prior_mode == 'kl':
				if have_prior and prior_alpha > 0.0 and prior is not None:
					log_p_tmp = torch.log_softmax(logit / tau, dim=-1)
					# 万一どちらかが非有限なら prior を無効化
					if (not torch.isfinite(log_p_tmp).all()) or (
						not torch.isfinite(prior).all()
					):
						if bool(getattr(cfg_fb, 'prior_debug', False)):
							print(
								'[trend prior KL] non-finite in prior/log_p -> disable prior for this batch'
							)
						prior_alpha = 0.0
						prior_ce = logit.new_tensor(0.0)
					else:
						kl_bh = -(prior * log_p_tmp).sum(dim=-1)
						use = (fb_idx >= 0) & covered
						prior_ce = kl_bh[use].mean() if use.any() else kl_bh.mean()
				else:
					prior_ce = logit.new_tensor(0.0)
			else:
				raise ValueError(f'Unknown prior_mode: {prior_mode}')

			# --- base KL over masked traces ----------------------------------
			base = fb_seg_kl_loss(
				logit.unsqueeze(1), target, valid_mask=valid_mask, tau=tau, eps=eps
			)
			prob = torch.softmax(logit / tau, dim=-1)  # (B,H,W)

			base_prior = (
				prior_alpha * prior_ce
				if (prior_mode == 'kl' and prior_alpha > 0)
				else base.new_tensor(0.0)
			)

			if (smooth_lambda <= 0 and smooth2_lambda <= 0) or offsets is None:
				total = base + base_prior
				return total, {
					'base': base.detach(),
					'smooth': base.new_tensor(0.0),
					'curv': base.new_tensor(0.0),
					'prior': base_prior.detach(),
					'prior_cov': covered.float().mean().detach(),
				}

			# --- position (sec) for smoothing terms --------------------------
			t = torch.arange(W, device=prob.device, dtype=prob.dtype)
			pos = (prob * t).sum(dim=-1)  # (B,H)
			if dt_sec is None:
				dt = pos.new_ones((pos.size(0), 1))
			else:
				dt = dt_sec.to(pos.device, pos.dtype).view(pos.size(0), 1)
			pos_sec = pos * dt  # (B,H)

			dx = (offsets[:, 1:] - offsets[:, :-1]).abs().to(pos_sec.dtype)
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

			total = (
				base + smooth_lambda * smooth + smooth2_lambda * loss_curv + base_prior
			)
			return total, {
				'base': base.detach(),
				'smooth': (smooth_lambda * smooth).detach(),
				'curv': (smooth2_lambda * loss_curv).detach(),
				'prior': base_prior.detach(),
				'prior_cov': covered.float().mean().detach(),
			}

		return _criterion

	if fb_type == 'mse':
		return make_fb_seg_mse_criterion()

	raise ValueError(f'Unknown fb_seg loss type: {fb_type}')


def make_fb_seg_mse_criterion():
	"""MSE criterion averaged over traces where fb_idx>=0."""

	def _criterion(
		pred: torch.Tensor,
		target: torch.Tensor,
		*,
		fb_idx: torch.Tensor,
		mask=None,
		**kwargs,
	) -> torch.Tensor:
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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .features import make_offset_channel
from .metrics import prepare_fb_windows, snr_improvement_from_cached_windows
from .predict import cover_all_traces_predict
from .velocity_mask import make_velocity_feasible_mask
from .vis import visualize_recon_triplet

__all__ = [
	'eval_synthe',
	'val_one_epoch_fbseg',
	'val_one_epoch_snr',
	'visualize_fb_seg_triplet',
]


def val_one_epoch_snr(
	model,
	val_loader,
	device,
	cfg_snr,
	visualize: bool = False,
	viz_batches: tuple[int, ...] = (0,),
	out_dir=None,
	writer=None,
	epoch: int | None = None,
	is_main_process: bool = True,
	use_offset_input: bool = False,
):
	"""Evaluate SNR improvement over validation loader."""
	import matplotlib.pyplot as plt

	model.eval()
	all_in, all_out, all_imp, all_vf = [], [], [], []
	for i, (x_masked, x_orig, _, meta) in enumerate(val_loader):
		x_orig = x_orig.to(device, non_blocking=True)
		fb_idx = meta['fb_idx'].to(device)
		offsets = None
		if use_offset_input:
			if 'offsets' not in meta:
				raise KeyError('offsets field is required when use_offset_input=True')
			offsets = meta['offsets']
		y_full = cover_all_traces_predict(
			model,
			x_orig,
			mask_ratio=cfg_snr.mask_ratio_for_eval,
			noise_std=getattr(cfg_snr, 'noise_std', 1.0),
			use_amp=True,
			device=device,
			seed=cfg_snr.seed,
			passes_batch=cfg_snr.passes_batch,
			mask_noise_mode=getattr(cfg_snr, 'mask_noise_mode', 'replace'),
			use_offset_input=use_offset_input,
			offsets=offsets,
		)
		cache = prepare_fb_windows(
			fb_idx,
			W=x_orig.shape[-1],
			pre_len=cfg_snr.pre_len,
			post_len=cfg_snr.post_len,
			guard=cfg_snr.guard,
		)
		res = snr_improvement_from_cached_windows(
			x_orig, y_full, cache, reduction='median'
		)
		all_in.append(res['snr_in_db'].item())
		all_out.append(res['snr_out_db'].item())
		all_imp.append(res['snr_improve_db'].item())
		all_vf.append(res['valid_frac'].item())
		if visualize and is_main_process and (i in viz_batches):
			gs = int(epoch) if isinstance(epoch, int) else 0
			fig = visualize_recon_triplet(
				x_orig,
				y_full,
				fb_idx=meta['fb_idx'],
				b=0,
				transpose=True,
				prefix=f'batch{i:04d}',
				writer=writer,
				global_step=gs,
			)
			plt.close(fig)
	return {
		'snr_in_db': float(np.median(all_in)),
		'snr_out_db': float(np.median(all_out)),
		'snr_improve_db': float(np.median(all_imp)),
		'valid_frac': float(np.mean(all_vf)),
	}


def eval_synthe(x_clean, pred, device=None):
	"""Compute MSE, MAE and PSNR for synthetic data."""
	mses, maes, psnrs = [], [], []
	for p, gt in zip(pred, x_clean, strict=False):
		if device is not None:
			p, gt = p.to(device), gt.to(device)
		mse = F.mse_loss(p, gt).item()
		mae = F.l1_loss(p, gt).item()
		psnr = -10.0 * torch.log10(F.mse_loss(p, gt)).item()
		mses.append(mse)
		maes.append(mae)
		psnrs.append(psnr)
	return {
		'mse': float(sum(mses) / len(mses)),
		'mae': float(sum(maes) / len(maes)),
		'psnr': float(sum(psnrs) / len(psnrs)),
		'num_shots': len(x_clean),
	}


def visualize_fb_seg_triplet(
	x, prob, fb_idx, *, b=0, writer=None, tag_prefix='fbseg', global_step=0
):
	"""Visualize input amplitude, probability heatmap, and overlay of predictions.

	Args:
		x: Input tensor of shape ``(B,1,H,W)``.
		prob: Probability tensor of shape ``(B,H,W)``.
		fb_idx: Ground-truth indices of shape ``(B,H)`` with ``-1`` as invalid.
		b: Batch index to visualize.
		writer: Optional TensorBoard writer.
		tag_prefix: Tag prefix for TensorBoard.
		global_step: Global step for TensorBoard.

	"""
	x_img = x[b, 0].detach().cpu().numpy()
	p = prob[b].detach().cpu().numpy()
	gt = fb_idx[b].detach().cpu().numpy()
	pred = p.argmax(axis=-1)
	h, _ = x_img.shape
	y = np.arange(h)

	fig, axes = plt.subplots(1, 3, figsize=(9, 4), dpi=200, constrained_layout=True)
	axes[0].imshow(
		x_img.T, aspect='auto', cmap='gray', interpolation='none', vmin=-3, vmax=3
	)
	axes[0].set_title('Amplitude')

	axes[1].imshow(
		p.T, aspect='auto', cmap='turbo', interpolation='none', vmin=0.0, vmax=1.0
	)
	axes[1].set_title('Probability')

	axes[2].imshow(
		x_img.T, aspect='auto', cmap='gray', interpolation='none', vmin=-3, vmax=3
	)
	axes[2].scatter(y, pred, s=2, c='lime', marker='o', label='Pred')
	valid = gt >= 0
	if valid.any():
		axes[2].scatter(
			y[valid], gt[valid], s=1, c='red', marker='o', label='GT', alpha=0.5
		)
	axes[2].set_title('Overlay')
	axes[2].legend(loc='upper right')

	W = x_img.shape[1]  # サンプル長（縦方向）
	valid = gt >= 0
	max_fb = int(gt[valid].max()) if valid.any() else int(pred.max())

	# 次の500の倍数に切り上げ（画像の高さは超えないようにクランプ）
	ymax = int(np.ceil((max_fb + 1) / 500.0) * 500)
	ymax = min(ymax, W - 1)

	# y軸: 上=0, 下=ymax。500サンプルごとに目盛り
	for ax in axes:
		ax.set_ylim(
			ymax,
			0,
		)  # 0 が上になる
		ax.set_yticks(np.arange(0, ymax + 1, 500))

	if writer is not None:
		writer.add_figure(f'{tag_prefix}/b{b:03d}', fig, global_step=global_step)
	return fig


@torch.no_grad()
def val_one_epoch_fbseg(
	model,
	val_loader,
	device,
	*,
	visualize=False,
	writer=None,
	epoch=0,
	viz_batches=(0,),
	cfg=None,
):
	"""Validate first-break segmentation model over one epoch.

	Returns:
		dict: Metrics containing hit@4, hit@8, and number of valid traces.

	"""
	model.eval()
	hit0 = 0
	hit2 = 0
	hit4 = 0
	hit8 = 0
	n_valid = 0
	cfg_fb = cfg.loss.fb_seg
	for i, (x, _, _, meta) in enumerate(val_loader):
		x = x.to(device, non_blocking=True)
		fb = meta['fb_idx'].to(device)
		x_in = x
		if (
			getattr(cfg, 'model', None)
			and getattr(cfg.model, 'use_offset_input', False)
			and ('offsets' in meta)
		):
			offs_ch = make_offset_channel(x, meta['offsets'])
			x_in = torch.cat([x, offs_ch], dim=1)
		logits = model(x_in)
		logit_raw = logits.squeeze(1)
		B, H, W = logit_raw.shape

		# === Always apply velocity-cone mask (match training) ===
		velmask = make_velocity_feasible_mask(
			offsets=meta['offsets'],
			dt_sec=meta['dt_sec'],
			W=W,
			vmin=float(getattr(cfg_fb, 'vmin_mask', 500.0)),
			vmax=float(getattr(cfg_fb, 'vmax_mask', 10000.0)),
			t0_lo_ms=float(getattr(cfg_fb, 't0_lo_ms', -100.0)),
			t0_hi_ms=float(getattr(cfg_fb, 't0_hi_ms', 80.0)),
			taper_ms=float(getattr(cfg_fb, 'taper_ms', 10.0)),
			device=logit_raw.device,
			dtype=logit_raw.dtype,
		)
		# fp32で安全にlogを作ってからdtypeを合わせる
		vm32 = velmask.to(torch.float32)
		has_any = vm32.sum(dim=-1, keepdim=True) > 0
		vm32 = torch.where(has_any, vm32, torch.ones_like(vm32))
		eps = float(getattr(cfg_fb, 'vel_log_eps', 1e-4))
		logmask32 = torch.log(vm32.clamp_min(eps))
		logit = logit_raw + logmask32.to(logit_raw.dtype)

		tau = float(getattr(cfg_fb, 'tau', 1.0))
		prob = torch.softmax(logit / tau, dim=-1)
		pred = prob.argmax(dim=-1)
		valid = fb >= 0
		diff = (pred - fb).abs()
		hit0 += ((diff == 0) & valid).sum().item()
		hit2 += ((diff <= 2) & valid).sum().item()
		hit4 += ((diff <= 4) & valid).sum().item()
		hit8 += ((diff <= 8) & valid).sum().item()
		n_valid += valid.sum().item()
		if visualize and (i in viz_batches):
			gs = int(epoch) if isinstance(epoch, int) else 0
			fig = visualize_fb_seg_triplet(
				x,
				prob,
				fb,
				b=0,
				writer=writer,
				tag_prefix=f'fbseg/batch{i:04d}',
				global_step=gs,
			)
			plt.close(fig)
	return {
		'hit@0': float(hit0) / max(n_valid, 1),
		'hit@2': float(hit2) / max(n_valid, 1),
		'hit@4': float(hit4) / max(n_valid, 1),
		'hit@8': float(hit8) / max(n_valid, 1),
		'n_tr_valid': int(n_valid),
	}

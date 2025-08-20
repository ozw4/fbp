import numpy as np
import torch
import torch.nn.functional as F
from metrics import prepare_fb_windows, snr_improvement_from_cached_windows
from vis import visualize_recon_triplet

from .predict import cover_all_traces_predict

__all__ = ['eval_synthe', 'val_one_epoch_snr']


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
):
	"""Evaluate SNR improvement over validation loader."""
	import matplotlib.pyplot as plt

	model.eval()
	all_in, all_out, all_imp, all_vf = [], [], [], []
	for i, (x_masked, x_orig, _, meta) in enumerate(val_loader):
		x_orig = x_orig.to(device, non_blocking=True)
		fb_idx = meta['fb_idx'].to(device)
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

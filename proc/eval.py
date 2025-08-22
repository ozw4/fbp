import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

__all__ = ['val_one_epoch_fbseg', 'visualize_fb_seg_triplet']


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
	axes[0].imshow(x_img, aspect='auto', cmap='gray', interpolation='none')
	axes[0].invert_yaxis()
	axes[0].set_title('Amplitude')

	axes[1].imshow(
		p, aspect='auto', cmap='turbo', interpolation='none', vmin=0.0, vmax=1.0
	)
	axes[1].invert_yaxis()
	axes[1].set_title('Probability')

	axes[2].imshow(x_img, aspect='auto', cmap='gray', interpolation='none')
	axes[2].scatter(pred, y, s=4, c='lime', marker='o', label='Pred')
	valid = gt >= 0
	if valid.any():
		axes[2].scatter(gt[valid], y[valid], s=4, c='red', marker='o', label='GT')
	axes[2].invert_yaxis()
	axes[2].set_title('Overlay')
	axes[2].legend(loc='upper right')

	if writer is not None:
		writer.add_figure(f'{tag_prefix}/b{b:03d}', fig, global_step=global_step)
	return fig


def val_one_epoch_fbseg(
	model,
	val_loader,
	device,
	*,
	visualize=False,
	writer=None,
	epoch=0,
	viz_batches=(0,),
):
	"""Validate first-break segmentation model over one epoch.

	Returns:
	    dict: Metrics containing hit@4, hit@8, and number of valid traces.

	"""
	model.eval()
	hit4 = 0
	hit8 = 0
	n_valid = 0
	for i, (x, _, _, meta) in enumerate(val_loader):
		x = x.to(device, non_blocking=True)
		fb = meta['fb_idx'].to(device)
		logits = model(x)
		prob = F.softmax(logits.squeeze(1), dim=-1)
		pred = prob.argmax(dim=-1)
		valid = fb >= 0
		diff = (pred - fb).abs()
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
		'hit@4': float(hit4) / max(n_valid, 1),
		'hit@8': float(hit8) / max(n_valid, 1),
		'n_tr_valid': int(n_valid),
	}

import time

import torch
import utils
from torch.amp.autocast_mode import autocast

from util.loss import shift_robust_l2_pertrace_vec


def _freeze_by_epoch(
	model: torch.nn.Module, epoch: int, freeze_epochs: int, unfreeze_steps: int
) -> None:
	"""Set ``requires_grad`` flags according to a freeze schedule.

	Parameters
	----------
	model: torch.nn.Module
	    Network with ``decoder`` and ``seg_head`` attributes.
	epoch: int
	    Current epoch index (0-based).
	freeze_epochs: int
	    Number of initial epochs during which only the head and the last
	    decoder blocks are trained.
	unfreeze_steps: int
	    After ``freeze_epochs`` epochs, additional decoder blocks are
	    unfrozen every ``unfreeze_steps`` epochs. Once all decoder blocks are
	    unfrozen, the encoder (pre/down/backbone) is unfrozen as well.

	"""
	if freeze_epochs <= 0 and unfreeze_steps <= 0:
		return

	decoder = getattr(model, 'decoder', None)
	seg_head = getattr(model, 'seg_head', None)

	if decoder is None or seg_head is None:
		return

	blocks = getattr(decoder, 'blocks', [])

	# Freeze everything first
	for p in model.parameters():
		p.requires_grad = False

	# Head is always trainable
	for p in seg_head.parameters():
		p.requires_grad = True

	if not blocks:
		return

	if epoch < freeze_epochs:
		start = max(len(blocks) - 2, 0)
	else:
		stage = (epoch - freeze_epochs) // max(1, unfreeze_steps)
		start = max(len(blocks) - (2 + stage), 0)

	# Unfreeze decoder blocks progressively (from last to first)
	for blk in blocks[start:]:
		for p in blk.parameters():
			p.requires_grad = True

	# When all decoder blocks are unfrozen, also unfreeze the encoder
	if start == 0 and epoch >= freeze_epochs:
		for attr in ('pre_down', 'extra_down', 'backbone'):
			module = getattr(model, attr, None)
			if module is not None:
				for p in module.parameters():
					p.requires_grad = True


__all__ = ['criterion', 'train_one_epoch']


def train_one_epoch(
	model,
	criterion,
	optimizer,
	lr_scheduler,
	dataloader,
	device,
	epoch,
	print_freq,
	writer=None,
	use_amp=True,
	scaler=None,
	ema=None,
	gradient_accumulation_steps=1,
	max_shift=5,
	step: int = 0,
	freeze_epochs: int = 0,
	unfreeze_steps: int = 1,
):
	"""Run one training epoch."""
	if getattr(model, '_transfer_loaded', False) and freeze_epochs > 0:
		_freeze_by_epoch(model, epoch, freeze_epochs, unfreeze_steps)

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(
		f'[epoch {epoch}] trainable: {trainable_params}/{total_params} '
		f'({trainable_params / total_params:.1%})'
	)
	model.train()
	metric_logger = utils.MetricLogger(delimiter='  ')
	metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
	metric_logger.add_meter(
		'samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}')
	)
        metric_logger.add_meter(
                'loss_base', utils.SmoothedValue(window_size=10, fmt='{value:.4f}')
        )
        metric_logger.add_meter(
                'loss_smooth', utils.SmoothedValue(window_size=10, fmt='{value:.4f}')
        )
        metric_logger.add_meter(
                'loss_curv', utils.SmoothedValue(window_size=10, fmt='{value:.4f}')
        )
	header = f'Epoch: [{epoch}]'
	optimizer.zero_grad()
	accum_loss = 0.0
        accum_loss_base = 0.0
        accum_loss_smooth = 0.0
        accum_loss_curv = 0.0
	for i, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
		x_masked, x_tgt, mask_or_none, meta = batch
		start_time = time.time()
		x_masked = x_masked.to(device, non_blocking=True)
		x_tgt = x_tgt.to(device, non_blocking=True)
		if mask_or_none is not None:
			mask_or_none = mask_or_none.to(device, non_blocking=True)
		meta = {
			k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
			for k, v in meta.items()
		}
		device_type = (
			'cuda' if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu'
		)
		with autocast(device_type=device_type, enabled=use_amp):
			pred = model(x_masked)
			out = criterion(
				pred,
				x_tgt,
				mask=mask_or_none,
				fb_idx=meta['fb_idx'],
				offsets=meta.get('offsets'),
			)
			if isinstance(out, tuple):
				total_loss, loss_logs = out
			else:
				total_loss, loss_logs = out, {}
			main_loss = total_loss / gradient_accumulation_steps
		if scaler:
			scaler.scale(main_loss).backward()
		else:
			main_loss.backward()
		accum_loss += main_loss.item()
                lb = loss_logs.get('base') if loss_logs else None
                ls = loss_logs.get('smooth') if loss_logs else None
                lc = loss_logs.get('curv') if loss_logs else None
                if lb is not None:
                        accum_loss_base += lb.item() / gradient_accumulation_steps
                if ls is not None:
                        accum_loss_smooth += ls.item() / gradient_accumulation_steps
                if lc is not None:
                        accum_loss_curv += lc.item() / gradient_accumulation_steps
		if (i + 1) % gradient_accumulation_steps == 0:
			if scaler:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
				scaler.step(optimizer)
				scaler.update()
			else:
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
				optimizer.step()
			if ema:
				ema.update(model)
			optimizer.zero_grad()
			metric_logger.update(loss=accum_loss, lr=optimizer.param_groups[0]['lr'])
			metric_logger.meters['samples/s'].update(
				x_masked.shape[0] / (time.time() - start_time)
			)
                        metric_logger.update(loss_base=accum_loss_base)
                        metric_logger.update(loss_smooth=accum_loss_smooth)
                        metric_logger.update(loss_curv=accum_loss_curv)
                        if writer:
                                writer.add_scalar('loss', accum_loss, step)
                                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
                                writer.add_scalar('loss_base', accum_loss_base, step)
                                writer.add_scalar('loss_smooth', accum_loss_smooth, step)
                                writer.add_scalar('loss_curv', accum_loss_curv, step)
                        step += 1
                        lr_scheduler.step()
                        accum_loss = 0.0
                        accum_loss_base = 0.0
                        accum_loss_smooth = 0.0
                        accum_loss_curv = 0.0

	return step


def criterion(
	pred: torch.Tensor,
	gt: torch.Tensor,
	*,
	mask: torch.Tensor | None = None,
	max_shift: int = 8,
	reduction: str = 'mean',
):
	"""Shift-robust L2 loss per trace."""
	return shift_robust_l2_pertrace_vec(
		pred, gt, mask=mask, max_shift=max_shift, reduction=reduction
	)

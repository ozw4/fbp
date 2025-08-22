import time

import torch
import utils
from torch.amp.autocast_mode import autocast

from util.loss import shift_robust_l2_pertrace_vec

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
):
	"""Run one training epoch."""
	model.train()
	metric_logger = utils.MetricLogger(delimiter='  ')
	metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
	metric_logger.add_meter(
		'samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}')
	)
	header = f'Epoch: [{epoch}]'
	optimizer.zero_grad()
	accum_loss = 0.0
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
			total_loss = criterion(
				pred, x_tgt, mask=mask_or_none, fb_idx=meta["fb_idx"]
			)
			main_loss = total_loss / gradient_accumulation_steps
		if scaler:
			scaler.scale(main_loss).backward()
		else:
			main_loss.backward()
		accum_loss += main_loss.item()
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
			if writer:
				writer.add_scalar('loss', accum_loss, step)
				writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
			step += 1
			lr_scheduler.step()
			accum_loss = 0.0

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

import time

import torch
import utils
from loss import shift_robust_l2_pertrace_vec
from torch.amp.autocast_mode import autocast

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
	# 追加：イテレーション損失用（常に更新してゼロ割を防ぐ）
	metric_logger.add_meter(
		'loss_iter', utils.SmoothedValue(window_size=10, fmt='{value:.4f}')
	)
	# 追加：更新ステップ毎の平均損失（任意）
	metric_logger.add_meter(
		'loss_step', utils.SmoothedValue(window_size=10, fmt='{value:.4f}')
	)

	header = f'Epoch: [{epoch}]'
	optimizer.zero_grad()
	accum_loss = 0.0
	accum_samples = 0

	device_type = (
		'cuda' if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu'
	)

	for i, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
		x_masked, x_orig, mask_2d, meta = batch
		start_time = time.time()
		x_masked = x_masked.to(device, non_blocking=True)
		x_orig = x_orig.to(device, non_blocking=True)
		mask_2d = mask_2d.to(device, non_blocking=True)

		with autocast(device_type=device_type, enabled=use_amp):
			pred = model(x_masked)
			total_loss = criterion(
				pred, x_orig, mask=mask_2d, max_shift=max_shift, reduction='mean'
			)
			main_loss = total_loss / gradient_accumulation_steps

		# 逆伝播
		if scaler:
			scaler.scale(main_loss).backward()
		else:
			main_loss.backward()

		accum_loss += main_loss.item()
		accum_samples += x_masked.size(0)

		# ★ここを「毎イテレーションで」更新（ゼロ割対策）
		metric_logger.update(
			loss_iter=total_loss.detach().item(),  # 生のイテレーション損失
			lr=optimizer.param_groups[0]['lr'],
		)
		metric_logger.meters['samples/s'].update(
			x_masked.size(0) / (time.time() - start_time)
		)

		# 勾配蓄積のステップ境界で最適化・ログ
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

			# ステップ平均の損失をここで記録（見やすさ用）
			step_loss = accum_loss  # すでに /grad_accum 済みを積んでいる
			metric_logger.update(loss_step=step_loss)

			if writer:
				writer.add_scalar('loss', step_loss, step)
				writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
				writer.add_scalar(
					'samples_per_sec', metric_logger.meters['samples/s'].value, step
				)

			step += 1
			lr_scheduler.step()

			accum_loss = 0.0
			accum_samples = 0


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

# %%
import copy
import datetime
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

from proc.eval import val_one_epoch_fbseg
from proc.util import utils
from proc.util.audit import audit_offsets_and_mask_coverage
from proc.util.collate import segy_collate
from proc.util.data_io import load_synth_pair
from proc.util.dataset import MaskedSegyGather
from proc.util.ema import ModelEMA
from proc.util.eval import eval_synthe, val_one_epoch_snr
from proc.util.loss import make_criterion, make_fb_seg_criterion
from proc.util.model import NetAE, adjust_first_conv_padding, inflate_first_conv_in
from proc.util.predict import cover_all_traces_predict_chunked
from proc.util.rng_util import worker_init_fn
from proc.util.train_loop import train_one_epoch
from proc.util.utils import WarmupCosineScheduler, collect_field_files, set_seed
from proc.util.vis import visualize_pair_quartet


def load_state_dict_excluding(
	model: torch.nn.Module,
	state: dict | str,
	exclude_prefixes: tuple[str, ...] = ('seg_head',),
	strict: bool = False,
):
	"""Load a checkpoint excluding parameters with given prefixes."""
	if isinstance(state, str):
		state = torch.load(state, map_location='cpu', weights_only=False)
	sd = state.get('model_ema', state)
	sd = {k: v for k, v in sd.items() if k.split('.')[0] not in exclude_prefixes}
	missing, unexpected = model.load_state_dict(sd, strict=strict)
	print(
		f'[transfer] loaded {len(sd)} keys; missing={len(missing)} unexpected={len(unexpected)}'
	)
	return missing, unexpected


SEED = 42
set_seed(SEED)
rng = np.random.default_rng()

with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='base')

if cfg.distributed:
	torch.cuda.set_device(cfg.local_rank)
	device = torch.device(cfg.device, cfg.local_rank)
else:
	device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
use_amp = cfg.use_amp
scaler = GradScaler(enabled=use_amp)

utils.init_distributed_mode(cfg)

# Select loss function based on task
criterion = make_criterion(cfg.loss)
task = getattr(cfg, 'task', 'recon')
if task == 'fb_seg':
	criterion = make_fb_seg_criterion(cfg.loss.fb_seg)

train_field_list = cfg.train_field_list
valid_field_list = cfg.valid_field_list
output_path = Path(f'./result/{task}/{train_field_list.split(".")[0]}_{cfg.suffix}')

utils.mkdir(output_path)
shutil.copy2('/workspace/proc/configs/base.yaml', output_path / 'config.yaml')


train_segy_files, train_fb_files = collect_field_files(
	cfg.train_field_list, cfg.data_root
)
valid_segy_files, valid_fb_files = collect_field_files(
	cfg.valid_field_list, cfg.data_root
)

train_dataset = MaskedSegyGather(
	train_segy_files,
	train_fb_files,
	mask_ratio=cfg.dataset.mask_ratio,
	mask_mode=cfg.dataset.mask_mode,
	mask_noise_std=cfg.dataset.mask_noise_std,
	target_mode=cfg.dataset.target_mode,
	label_sigma=cfg.dataset.label_sigma,
	flip=cfg.dataset.flip,
	augment_time_prob=cfg.dataset.augment.time.prob,
	augment_time_range=tuple(cfg.dataset.augment.time.range),
	augment_space_prob=cfg.dataset.augment.space.prob,
	augment_space_range=tuple(cfg.dataset.augment.space.range),
	augment_freq_prob=cfg.dataset.augment.freq.prob,
	augment_freq_kinds=tuple(cfg.dataset.augment.freq.kinds),
	augment_freq_band=tuple(cfg.dataset.augment.freq.band),
	augment_freq_width=tuple(cfg.dataset.augment.freq.width),
	augment_freq_roll=cfg.dataset.augment.freq.roll,
	augment_freq_restandardize=cfg.dataset.augment.freq.restandardize,
)
if task == 'fb_seg':
	valid_dataset = MaskedSegyGather(
		valid_segy_files,
		valid_fb_files,
		mask_ratio=0,
		mask_mode=cfg.dataset.mask_mode,
		mask_noise_std=0,
		target_mode=cfg.dataset.target_mode,
		label_sigma=cfg.dataset.label_sigma,
		flip=False,
		augment_time_prob=0.0,
		augment_space_prob=0.0,
		augment_freq_prob=0.0,
	)
elif task == 'recon':
	valid_dataset = MaskedSegyGather(
		valid_segy_files,
		valid_fb_files,
		mask_ratio=cfg.dataset.mask_ratio,
		mask_mode=cfg.dataset.mask_mode,
		mask_noise_std=cfg.dataset.mask_noise_std,
		target_mode=cfg.dataset.target_mode,
		label_sigma=cfg.dataset.label_sigma,
		flip=False,
		augment_time_prob=0.0,
		augment_space_prob=0.0,
		augment_freq_prob=0.0,
	)
val_src = copy.copy(valid_dataset)  # file_infos を共有
val_src.flip = False  # ここだけ無反転で取りたい場合

n_val_item = (
	cfg.val_steps * cfg.batch_size
	if not cfg.distributed
	else cfg.val_steps * cfg.batch_size * cfg.world_size
)
val_items = [val_src[i] for i in range(n_val_item)]
val_src.close()


class FrozenValDataset(Dataset):
	def __init__(self, items):
		self.items = items

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i):
		return self.items[i]


valid_dataset = FrozenValDataset(val_items)

train_writer = SummaryWriter(os.path.join(output_path, 'logs', 'train'))
valid_writer = SummaryWriter(os.path.join(output_path, 'logs', 'val'))

if cfg.distributed:
	train_sampler = DistributedSampler(
		train_dataset,
		shuffle=False,  # Dataset内部がランダム生成なのでshuffle不要
		drop_last=True,
	)
else:
	# 無限サンプリング的にreplacement=Trueにしてステップ数制御も可能
	train_sampler = RandomSampler(
		train_dataset,
		replacement=True,  # for using num_samples
		num_samples=cfg.steps_per_epoch * cfg.batch_size,  # 必要に応じて制御
	)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
	train_dataset,
	batch_size=cfg.batch_size,  # 固定4はやめてcfgから
	sampler=train_sampler,  # ★samplerを使う
	shuffle=False,  # ★sampler併用時はFalse
	num_workers=cfg.num_workers,  # ここもcfgから
	pin_memory=True,
	collate_fn=segy_collate,  # ★必須
	worker_init_fn=worker_init_fn,
	drop_last=True,
	generator=g,
)

if cfg.distributed:
	n_val_item = cfg.val_steps * cfg.batch_size * cfg.world_size
else:
	n_val_item = cfg.val_steps * cfg.batch_size


val_loader = DataLoader(
	valid_dataset,
	batch_size=cfg.batch_size,
	sampler=SequentialSampler(valid_dataset),  # または shuffle=False
	shuffle=False,
	num_workers=0,  # 読むだけなので 0 で十分（>0でもOK）
	pin_memory=True,
	collate_fn=segy_collate,
	drop_last=False,
	worker_init_fn=worker_init_fn,
)

# -------- DEBUG AUDIT (run once before training) --------
try:
	audit_batches = int(getattr(getattr(cfg, 'debug', object()), 'audit_batches', 50))
	cov_th = float(getattr(getattr(cfg, 'debug', object()), 'audit_cov_threshold', 0.5))
except Exception:
	audit_batches, cov_th = 50, 0.5

if audit_batches > 0:
	print(
		f'[AUDIT] running audit on {audit_batches} batches (coverage threshold={cov_th})'
	)
	audit_offsets_and_mask_coverage(
		train_loader, cfg.loss.fb_seg, max_batches=audit_batches, cov_threshold=cov_th
	)
	# Optionally also audit the validation loader:
	# audit_offsets_and_mask_coverage(val_loader, cfg.loss.fb_seg, max_batches=min(20, audit_batches), cov_threshold=cov_th)
print('[AUDIT] done.')

synthe_noise_segy = (
	'/home/dcuser/data/Synthetic/marmousi/shot801_decimate_fieldnoise008.sgy'
)
synthe_clean_segy = '/home/dcuser/data/Synthetic/marmousi/shotdata001_801_decimate.sgy'

# 例) FFID 401 と 601 を抽出（W は 6016 に揃える）
synthe_noisy, synthe_clean, _, used_ffids, Hs = load_synth_pair(
	synthe_noise_segy,
	synthe_clean_segy,
	extract_key1idxs=[1, 401, 801, 1201, 1601],
	target_len=6016,
	standardize=True,
	endian='little',
)

print(cfg.backbone)
model = NetAE(
		backbone=cfg.backbone,
		pretrained=True,
		stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
		pre_stages=2,
		pre_stage_strides=(
				(1, 1),
				(1, 2),
		),
).to(device)

if getattr(cfg.model, 'use_offset_input', False):
		inflate_first_conv_in(model, in_ch=2)

if 'caformer' in cfg.backbone:
		adjust_first_conv_padding(model.backbone, padding=(3, 3))

if cfg.distributed and cfg.sync_bn:
	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

epochs = cfg.epoch_block * cfg.num_block
lr = cfg.lr * cfg.world_size
step = 0

base_lr = cfg.lr * cfg.world_size
param_groups = [
	{'params': model.seg_head.parameters(), 'lr': base_lr},
]
if hasattr(model, 'decoder'):
	param_groups.append({'params': model.decoder.parameters(), 'lr': base_lr * 0.5})

enc_params = [
	p
	for n, p in model.named_parameters()
	if not n.startswith('seg_head') and not n.startswith('decoder')
]
if enc_params:
	param_groups.append({'params': enc_params, 'lr': base_lr * 0.1})

optimizer = torch.optim.AdamW(
	param_groups,
	betas=(0.9, 0.999),
	weight_decay=cfg.weight_decay,
)

warmup_iters = cfg.lr_warmup_epochs * len(train_loader)

lr_scheduler = WarmupCosineScheduler(
	optimizer=optimizer,
	warmup_steps=cfg.lr_warmup_epochs * len(train_loader),
	total_steps=epochs * len(train_loader),
	eta_min=1e-6,
)

model_without_ddp = model

if cfg.distributed:
	model = DistributedDataParallel(model, device_ids=[cfg.local_rank])
	model_without_ddp = model.module

transfer_loaded = False
if cfg.resume:
	transfer_loaded = True
	checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=False)

	# 1) モデル重みを読み込む（必要ならヘッドを除外）
	if getattr(cfg, 'resume_exclude_head', False):
		prefixes = tuple(getattr(cfg, 'resume_exclude_prefixes', ('seg_head',)))
		load_state_dict_excluding(
			model_without_ddp, checkpoint, exclude_prefixes=prefixes, strict=False
		)
	else:
		model_state = checkpoint.get('model_ema', checkpoint)
		model_without_ddp.load_state_dict(model_state, strict=False)

	# 2) optimizer / lr_scheduler は基本読まない（明示フラグがある時のみ試す）
	if getattr(cfg, 'resume_load_optim', False):
		try:
			opt_sd = checkpoint['optimizer']
			if len(opt_sd['param_groups']) == len(optimizer.param_groups):
				optimizer.load_state_dict(opt_sd)
				if 'lr_scheduler' in checkpoint:
					lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
				cfg.start_epoch = int(checkpoint.get('epoch', -1)) + 1
				step = int(checkpoint.get('step', 0))
			else:
				raise ValueError('param_groups mismatch')
		except Exception as e:
			print(
				f'[resume] skip optimizer/scheduler: {e} -> reinit and restart from epoch 0'
			)
	else:
		print('[resume] loaded model weights only (optimizer/scheduler reinitialized)')

# 3) 段階解凍のガード用フラグ
model._transfer_loaded = transfer_loaded

if not cfg.distributed:
	model = torch.compile(model, fullgraph=True, dynamic=False, mode='default')

ema = (
	ModelEMA(model_without_ddp, decay=cfg.ema_decay, device=device)
	if cfg.use_ema
	else None
)

if cfg.resume and ema:
	print('Loading EMA model from checkpoint')
	model_ema_state = checkpoint['model_ema']
	ema.load_state_dict(model_ema_state, strict=False)

eval_model = ema.module if ema else model

print('Start training')
start_time = time.time()
best_mse = checkpoint.get('best_mse', float('inf')) if cfg.resume else float('inf')
best_snr = checkpoint.get('best_snr', float('-inf')) if cfg.resume else float('-inf')
chp = 1
step = checkpoint.get('step', 0) if cfg.resume else 0

for epoch in range(cfg.start_epoch, epochs):
	if cfg.distributed:
		train_sampler.set_epoch(epoch)

		step = train_one_epoch(
				model=model,
				criterion=criterion,
				optimizer=optimizer,
				lr_scheduler=lr_scheduler,
				dataloader=train_loader,
				device=device,
				epoch=epoch,
				print_freq=cfg.print_freq,
				writer=train_writer,
				use_amp=use_amp,
				scaler=scaler,
				ema=ema,
				gradient_accumulation_steps=1,
				step=step,
				freeze_epochs=cfg.freeze_epochs,
				unfreeze_steps=cfg.unfreeze_steps,
				use_offset_input=getattr(cfg.model, 'use_offset_input', False),
		)
	eval_model = ema.module if ema else model

	if task == 'recon':
				snr_dict = val_one_epoch_snr(
						eval_model,
						val_loader,
						device=device,
						cfg_snr=cfg.snr,
						visualize=True,
						writer=valid_writer,
						epoch=epoch,
						is_main_process=utils.is_main_process(),
						viz_batches=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
						if utils.is_main_process()
						else (),
						use_offset_input=getattr(cfg.model, 'use_offset_input', False),
				)

		# 合成データ推論 & 指標
				pred = cover_all_traces_predict_chunked(
						eval_model,
						synthe_noisy.to(device),
						mask_noise_mode=cfg.dataset.mask_mode,
						noise_std=cfg.dataset.mask_noise_std,
						use_offset_input=getattr(cfg.model, 'use_offset_input', False),
				)
		synthe_metrics = eval_synthe(synthe_clean, pred, device=device)
		for i in range(len(synthe_noisy)):
			visualize_pair_quartet(
				synthe_noisy[:, :, :, :1251],
				pred[:, :, :, :1251],
				synthe_clean[:, :, :, :1251],
				b=i,  # 1ショットのみ
				transpose=True,
				prefix='synth',
				writer=valid_writer,
				global_step=epoch,
			)

		# ── TensorBoard: エポック単位で記録 ──
		if utils.is_main_process():
			valid_writer.add_scalars(
				'val_snr',
				{
					#'in_db': snr_dict['snr_in_db'],
					'out_db': snr_dict['snr_out_db'],
					#'improve_db': snr_dict['snr_improve_db'],
				},
				epoch,
			)
			# valid_writer.add_scalar('val_snr/valid_frac', snr_dict['valid_frac'], epoch)

			valid_writer.add_scalar('val_synth/mse', synthe_metrics['mse'], epoch)
			valid_writer.add_scalar('val_synth/mae', synthe_metrics['mae'], epoch)
			valid_writer.add_scalar('val_synth/psnr', synthe_metrics['psnr'], epoch)
			valid_writer.flush()

		# ── ベスト更新（MSEは小さいほど良い / SNR改善は大きいほど良い） ──
		save_flag = False
		if synthe_metrics['mse'] < best_mse:
			best_mse = synthe_metrics['mse']
			save_flag = True
		if snr_dict['snr_improve_db'] > best_snr:
			best_snr = snr_dict['snr_improve_db']
			save_flag = True

		# チェックポイント作成
		checkpoint = {
			'model': model_without_ddp.state_dict(),
			'optimizer': optimizer.state_dict(),
			'lr_scheduler': lr_scheduler.state_dict(),
			'epoch': epoch,
			'step': step,
			'cfg': cfg,
			'best_mse': best_mse,
			'best_snr': best_snr,
			'rng_state': {
				'torch': torch.get_rng_state(),
				'cuda': torch.cuda.get_rng_state_all(),
				'numpy': np.random.get_state(),
				'random': random.getstate(),
			},
		}
		if ema:
			checkpoint['model_ema'] = ema.module.state_dict()

		if save_flag:
			utils.save_on_master(
				checkpoint, os.path.join(output_path, 'checkpoint.pth')
			)
			print('saving checkpoint at epoch:', epoch)

		print('epoch:', epoch)
		print(
			'  SNR in/out/Δ(dB):',
			snr_dict['snr_in_db'],
			snr_dict['snr_out_db'],
			snr_dict['snr_improve_db'],
		)
		print('  best SNR Δ(dB):', best_snr)
		print('  synth MSE/MAE/PSNR:', synthe_metrics)
		print('  best MSE:', best_mse)
	elif task == 'fb_seg':
		fb_metrics = val_one_epoch_fbseg(
			eval_model,
			val_loader,
			device=device,
			visualize=True,
			writer=valid_writer,
			epoch=epoch,
			viz_batches=(0, 1, 2, 3, 4),
			cfg=cfg,
		)
		if utils.is_main_process():
			valid_writer.add_scalar('fbseg/hit_at_0', fb_metrics['hit@0'], epoch)
			valid_writer.add_scalar('fbseg/hit_at_2', fb_metrics['hit@2'], epoch)
			valid_writer.add_scalar('fbseg/hit_at_4', fb_metrics['hit@4'], epoch)
			valid_writer.add_scalar('fbseg/hit_at_8', fb_metrics['hit@8'], epoch)
			valid_writer.add_scalar('fbseg/n_tr_valid', fb_metrics['n_tr_valid'], epoch)
			valid_writer.flush()
		checkpoint = {
			'model': model_without_ddp.state_dict(),
			'optimizer': optimizer.state_dict(),
			'lr_scheduler': lr_scheduler.state_dict(),
			'epoch': epoch,
			'step': step,
			'cfg': cfg,
			'rng_state': {
				'torch': torch.get_rng_state(),
				'cuda': torch.cuda.get_rng_state_all(),
				'numpy': np.random.get_state(),
				'random': random.getstate(),
			},
		}
		if ema:
			checkpoint['model_ema'] = ema.module.state_dict()
		print('epoch:', epoch)
	else:
		checkpoint = {
			'model': model_without_ddp.state_dict(),
			'optimizer': optimizer.state_dict(),
			'lr_scheduler': lr_scheduler.state_dict(),
			'epoch': epoch,
			'step': step,
			'cfg': cfg,
			'rng_state': {
				'torch': torch.get_rng_state(),
				'cuda': torch.cuda.get_rng_state_all(),
				'numpy': np.random.get_state(),
				'random': random.getstate(),
			},
		}
		if ema:
			checkpoint['model_ema'] = ema.module.state_dict()
		print('epoch:', epoch)

	if output_path and (epoch + 1) % cfg.epoch_block == 0:
		utils.save_on_master(
			checkpoint, os.path.join(output_path, f'model_{epoch + 1}.pth')
		)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f'Training time {total_time_str}')

# %%

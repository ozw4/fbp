# %%
import copy
import datetime
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import utils
from ema import ModelEMA
from hydra import compose, initialize
from model import NetAE, adjust_first_conv_padding
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from util import (
	MaskedSegyGather,
	cover_all_traces_predict_chunked,
	criterion,
	eval_synthe,
	load_synth_pair,
	segy_collate,
	train_one_epoch,
	val_one_epoch_snr,
	worker_init_fn,
)
from utils import WarmupCosineScheduler, set_seed
from vis import visualize_pair_quartet

SEED = 42
set_seed(SEED)
rng = np.random.default_rng()

with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='train')
if cfg.distributed:
	torch.cuda.set_device(cfg.local_rank)
	device = torch.device(cfg.device, cfg.local_rank)
else:
	device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
use_amp = cfg.use_amp
scaler = GradScaler(enabled=use_amp)

utils.init_distributed_mode(cfg)

train_field_list = cfg.train_field_list
with open(f'/workspace/proc/configs/{train_field_list}') as f:
	field_list = f.read().splitlines()

segy_files = []
fb_files = []
for field in field_list:
	# SEG-YとFBファイルを取得
	field_dir = Path(cfg.data_root) / field
	segy_file = list(field_dir.glob('*.sgy'))
	fb_file = list(field_dir.glob('*.npy'))
	if len(segy_file) != 1 or len(fb_file) != 1:
		print(f'No SEG-Y or FB files found in {field}')
		continue
	segy_files.extend(segy_file)
	fb_files.extend(fb_file)

train_dataset = MaskedSegyGather(
	segy_files,
	fb_files,
	mask_ratio=0.5,
	flip=True,
	augment_time_prob=0.3,
	augment_time_range=(0.8, 1.2),
	augment_space_prob=0.3,
	augment_space_range=(0.8, 1.2),
	augment_freq_prob=0.5,  # ← 追加
	augment_freq_kinds=('bandpass', 'lowpass', 'highpass'),
	augment_freq_band=(0.05, 0.45),
	augment_freq_width=(0.12, 0.35),
	augment_freq_roll=0.02,
	augment_freq_restandardize=True,
)
valid_dataset = MaskedSegyGather(
	segy_files,
	fb_files,
	mask_ratio=0.5,
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


class FrozenValDataset(Dataset):
	def __init__(self, items):
		self.items = items

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i):
		return self.items[i]


valid_dataset = FrozenValDataset(val_items)
output_path = Path(f'./result/{train_field_list.split(".")[0]}_{cfg.suffix}')

utils.mkdir(output_path)
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

model = NetAE(
	backbone='caformer_b36.sail_in22k_ft_in1k',
	pretrained=True,
	stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
	pre_stages=2,
	pre_stage_strides=(
		(1, 1),
		(1, 2),
	),
).to(device)
adjust_first_conv_padding(model.backbone, padding=(3, 3))

if cfg.distributed and cfg.sync_bn:
	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

epochs = cfg.epoch_block * cfg.num_block
lr = cfg.lr * cfg.world_size
step = 0

optimizer = torch.optim.AdamW(
	model.parameters(),
	lr=cfg.lr * cfg.world_size,
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

if cfg.resume:
	checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=False)
	model_without_ddp.load_state_dict(
		checkpoint['model'], strict=False
	)  # strict=False to allow partial loading
	optimizer.load_state_dict(checkpoint['optimizer'])
	lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
	cfg.start_epoch = checkpoint['epoch'] + 1
	step = checkpoint['step']

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

	train_one_epoch(
		model=model,
		criterion=criterion,
		optimizer=optimizer,
		lr_scheduler=lr_scheduler,
		dataloader=train_loader,  # ★ここをtrain_datasetから修正
		device=device,
		epoch=epoch,
		print_freq=cfg.print_freq,
		writer=train_writer,
		use_amp=use_amp,
		scaler=scaler,
		ema=ema,
		gradient_accumulation_steps=1,
		step=step,
	)
	eval_model = ema.module if ema else model

	snr_dict = val_one_epoch_snr(
		eval_model,
		val_loader,
		device=device,
		cfg_snr=cfg.snr,
		visualize=True,
		writer=valid_writer,
		epoch=epoch,
		is_main_process=utils.is_main_process(),
		viz_batches=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9) if utils.is_main_process() else (),
	)

	# 合成データ推論 & 指標
	pred = cover_all_traces_predict_chunked(eval_model, synthe_noisy.to(device))
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
				'in_db': snr_dict['snr_in_db'],
				'out_db': snr_dict['snr_out_db'],
				'improve_db': snr_dict['snr_improve_db'],
			},
			epoch,
		)
		valid_writer.add_scalar('val_snr/valid_frac', snr_dict['valid_frac'], epoch)

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
		utils.save_on_master(checkpoint, os.path.join(output_path, 'checkpoint.pth'))
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
	if output_path and (epoch + 1) % cfg.epoch_block == 0:
		utils.save_on_master(
			checkpoint, os.path.join(output_path, f'model_{epoch + 1}.pth')
		)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f'Training time {total_time_str}')

# %%

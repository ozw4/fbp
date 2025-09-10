# %%
# debug_prior_visual.py
# build_val_loader_and_model.py
import copy

# debug_prior_visual_sections.py
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from proc.util import MaskedSegyGather, segy_collate, worker_init_fn
from proc.util.features import make_offset_channel
from proc.util.model_utils import inflate_input_convs_to_2ch

# loss.py からロバスト回帰だけ借ります
from proc.util.loss import gaussian_prior_from_trend, robust_linear_trend_sections
from proc.util.model import NetAE, adjust_first_conv_padding


@torch.no_grad()
def debug_prior_visual(
	model,
	loader,
	device,
	cfg_fb,
	outdir='debug_prior_sections',
	num_traces_to_plot=6,
	max_sections=20,  # 走査するセクション数（アイテム数）の上限
	section_stride=1,  # 等間引き（例: 5 なら 0,5,10,... を処理）
):
	"""DataLoader の“各セクション（=各アイテム）”単位で:
	  推論 -> pos_sec -> ロバスト回帰 -> ガウスprior -> 数値診断/可視化 を実行。

	生成物:
	  - outdir/section_XXXXX/trace_prob_vs_prior.png
	  - outdir/section_XXXXX/trend_over_offsets.png
	  - outdir/section_XXXXX/vtrend_over_offsets.png
	  - outdir/summary.csv（residual/CE の要約）
	"""
	model.eval()
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)

	tau = float(getattr(cfg_fb, 'tau', 1.0))
	prior_sigma_ms = float(getattr(cfg_fb, 'prior_sigma_ms', 20.0))
	section_len = int(getattr(cfg_fb, 'trend_section', 128))
	stride = int(getattr(cfg_fb, 'trend_stride', 64))
	huber_c = float(getattr(cfg_fb, 'trend_huber_c', 1.345))
	vmin = float(getattr(cfg_fb, 'trend_vmin', 500.0))
	vmax = float(getattr(cfg_fb, 'trend_vmax', 5000.0))

	# サマリ CSV
	csv_path = outdir / 'summary.csv'
	with open(csv_path, 'w', newline='') as fcsv:
		writer = csv.writer(fcsv)
		writer.writerow(
			[
				'section_idx',
				'batch_idx',
				'item_in_batch',
				'B',
				'H',
				'W',
				'sigma_ms',
				'res_ms_median',
				'res_ms_p95',
				'res_ms_p99',
				'ce_prior_p_mean',
			]
		)

		section_counter = 0
		for batch_idx, batch in enumerate(loader):
			if (batch_idx % section_stride) != 0:
				continue
			if section_counter >= max_sections:
				break

                        x_masked, x_tgt, mask_or_none, meta = batch
                        x_masked = x_masked.to(device, non_blocking=True)
                        x_in = x_masked
                        if getattr(cfg, 'model', None) and getattr(cfg.model, 'use_offset_input', False) and ('offsets' in meta):
                                offs_ch = make_offset_channel(x_masked, meta['offsets'])
                                x_in = torch.cat([x_masked, offs_ch], dim=1)
                        logits = model(x_in)  # (B,1,H,W) or (B,2,H,W)
                        logit = logits.squeeze(1)  # (B,H,W)
                        prob = torch.softmax(logit / tau, dim=-1)  # (B,H,W)
                        B, H, W = prob.shape
			t_idx = torch.arange(W, device=prob.device, dtype=prob.dtype).view(1, 1, W)

			dt_sec = meta['dt_sec'].to(prob).view(B, 1)  # (B,1)
			offsets = meta['offsets'].to(prob)  # (B,H)
			fb_idx = meta['fb_idx'].to(torch.long)  # (B,H)
			valid = fb_idx > 0

			# 期待位置（秒）
			pos_samples = (prob * t_idx).sum(dim=-1)  # (B,H)
			pos_sec = pos_samples * dt_sec  # (B,H)

			# バッチ内の各アイテム（=各セクション）を個別に可視化する
			for b in range(B):
				sec_dir = outdir / f'section_{section_counter:05d}'
				sec_dir.mkdir(parents=True, exist_ok=True)

				# ---- ロバスト回帰（このアイテムのみ）----
				t_tr, s_tr, v_tr, w_conf, covered = robust_linear_trend_sections(
					offsets=offsets[b : b + 1],
					t_sec=pos_sec[b : b + 1],
					valid=valid[b : b + 1],
					prob=prob[b : b + 1],  # ← prob を渡す
					dt_sec=dt_sec[b : b + 1],  # ← dt を渡す
					section_len=section_len,
					stride=stride,
					huber_c=huber_c,
					iters=3,
					vmin=vmin,
					vmax=vmax,
				)  # 返りは (1,H)

				# ---- prior 生成（このアイテムのみ）----
				prior = gaussian_prior_from_trend(
					t_trend_sec=t_tr,
					dt_sec=dt_sec[b : b + 1],
					W=W,
					sigma_ms=prior_sigma_ms,
					ref_tensor=logits[b : b + 1],
					covered_mask=covered,  # ← 追加
				)  # (1,H,W)

				# ---- 数値診断 ----
				res = (pos_sec[b : b + 1] - t_tr).abs()  # (1,H)
				res_valid = res[valid[b : b + 1]]
				if res_valid.numel() > 0:
					res_ms = (res_valid * 1e3).cpu()
					med_ms = float(res_ms.median())
					p95_ms = float(res_ms.quantile(0.95))
					p99_ms = float(res_ms.quantile(0.99))
				else:
					med_ms = p95_ms = p99_ms = float('nan')
				covered = covered.to(fb_idx.device)
				log_p = torch.log_softmax(logit[b : b + 1] / tau, dim=-1)  # (1,H,W)
				ce_bh = -(prior * log_p).sum(dim=-1)  # (1,H)

				cov = covered[0]  # (H,)
				use = (fb_idx[b] >= 0) & cov  # ← デバイス一致済み
				ce = float(ce_bh[0, use].mean().cpu()) if use.any() else float('nan')

				print(
					f'[prior dbg] sec={section_counter:05d} (batch={batch_idx}, item={b}) '
					f'B=1 H={H} W={W}  sigma={prior_sigma_ms}ms  '
					f'|pos-trend|ms: med={med_ms:.3f} p95={p95_ms:.3f} p99={p99_ms:.3f}  '
					f'CE={ce:.5f}'
				)
				writer.writerow(
					[
						section_counter,
						batch_idx,
						b,
						1,
						H,
						W,
						prior_sigma_ms,
						med_ms,
						p95_ms,
						p99_ms,
						ce,
					]
				)

				# ---- 可視化（このアイテムのみ）----
				# 1) トレース上の p(t) と prior(t)
				valid_idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
				if valid_idx.numel() > 0:
					k = min(num_traces_to_plot, int(valid_idx.numel()))
					take = (
						torch.linspace(0, valid_idx.numel() - 1, steps=k).round().long()
					)
					traces = valid_idx[take].tolist()

					t_s = (
						(t_idx * dt_sec[b : b + 1].view(1, 1)).squeeze().cpu().numpy()
					)  # (W,)
					fig = plt.figure(figsize=(12, 2.6 * k))
					for j, h in enumerate(traces, start=1):
						ax = fig.add_subplot(k, 1, j)
						ax.plot(
							t_s, prob[b, h].cpu().numpy(), label=f'prob (trace {h})'
						)
						ax.plot(
							t_s,
							prior[0, h].cpu().numpy(),
							label=f'prior (σ={prior_sigma_ms}ms)',
						)
						ax.axvline(
							float(t_tr[0, h].cpu()),
							linestyle='--',
							linewidth=1,
							label='trend μ',
						)
						ax.set_xlim(t_s[0], t_s[-1])
						ax.set_xlabel('time [s]')
						ax.set_ylabel('prob')
						ax.legend(loc='best')
					fig.tight_layout()
					fig.savefig(sec_dir / 'trace_prob_vs_prior.png', dpi=150)
					plt.close(fig)

				ord_idx = torch.argsort(offsets[b])
				x_np = offsets[b].detach().cpu().numpy()
				pos_np = pos_sec[b].detach().cpu().numpy()
				tt_np = t_tr[0].detach().cpu().numpy()
				valid_np = valid[b].detach().cpu().numpy().astype(bool)

				# オフセット昇順に並べ替え（numpy で統一）
				ord_idx = np.argsort(x_np)
				x_plot = x_np[ord_idx]
				pos_plot = pos_np[ord_idx]
				tt_plot = tt_np[ord_idx]
				valid_plot = valid_np[ord_idx]
				# 無効トレースは線を切る（NaN にする）
				tt_plot_masked = tt_plot.copy()
				tt_plot_masked[~valid_plot] = np.nan

				fig2 = plt.figure(figsize=(10, 4))
				ax2 = fig2.add_subplot(1, 1, 1)
				# pos_sec は有効だけ散布
				ax2.plot(x_plot[valid_plot], pos_plot[valid_plot], '.', label='pos_sec')
				# trend_t は NaN で分断された線として描く
				cov = covered[0].cpu().numpy().astype(bool)
				tt_plot = t_tr[0].cpu().numpy()
				tt_plot[~cov] = float('nan')
				ax2.plot(offsets[b].cpu().numpy(), tt_plot, '-', label='trend_t (fit)')
				ax2.set_xlabel('offset [m]')
				ax2.set_ylabel('time [s]')
				ax2.legend(loc='best')
				fig2.tight_layout()
				fig2.savefig(sec_dir / 'trend_over_offsets.png', dpi=150)
				plt.close(fig2)

				# --- 3) 見かけ速度 v_trend ---
				v_plot = (1.0 / s_tr.clamp_min(1e-6))[0][ord_idx].cpu().numpy()
				v_plot_masked = v_plot.copy()
				v_plot_masked[~valid_plot] = np.nan

				fig3 = plt.figure(figsize=(10, 3))
				ax3 = fig3.add_subplot(1, 1, 1)
				ax3.plot(x_plot, v_plot_masked, '-')
				ax3.set_xlabel('offset [m]')
				ax3.set_ylabel('v_trend [m/s]')
				fig3.tight_layout()
				fig3.savefig(sec_dir / 'vtrend_over_offsets.png', dpi=150)
				plt.close(fig3)

				section_counter += 1
				if section_counter >= max_sections:
					break

	print(f'[prior dbg] saved summary to: {csv_path}')


# build_val_loader_and_model.py

import torch

# ---- 1) Config / device ----
with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='base')

device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')


# ---- 2) フィールドごとに .sgy/.npy を収集（最小版）----
def collect_field_files(list_name: str, data_root: str):
	list_path = Path('configs') / list_name
	with open(list_path) as f:
		fields = [
			ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')
		]
	segy_files, fb_files = [], []
	for field in fields:
		d = Path(data_root) / field
		segy = sorted(list(d.glob('*.sgy')) + list(d.glob('*.segy')))
		fb = sorted(d.glob('*.npy'))
		if not segy or not fb:
			print(f'[warn] skip: {field}')
			continue
		segy_files.append(segy[0])
		fb_files.append(fb[0])
	if not segy_files:
		raise RuntimeError(f'No usable fields in {list_name}')
	return segy_files, fb_files


_, valid_fb_files = collect_field_files(cfg.valid_field_list, cfg.data_root)
valid_segy_files, _ = collect_field_files(cfg.valid_field_list, cfg.data_root)

# ---- 3) 検証用データセット（Augなし・flipなし）----
valid_dataset_full = MaskedSegyGather(
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


# 「サンプル数を固定したい」場合は最初の n アイテムに凍結
class FrozenValDataset(Dataset):
	def __init__(self, items):
		self.items = items

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i):
		return self.items[i]


n_val_item = cfg.val_steps * cfg.batch_size
val_src = copy.copy(valid_dataset_full)  # file_infos を共有
val_src.flip = False
val_items = [val_src[i] for i in range(n_val_item)]
valid_dataset = FrozenValDataset(val_items)

# ---- 4) DataLoader（検証用）----
val_loader = DataLoader(
	valid_dataset,
	batch_size=cfg.batch_size,
	sampler=SequentialSampler(valid_dataset),
	shuffle=False,
	num_workers=0,
	pin_memory=True,
	collate_fn=segy_collate,
	drop_last=False,
	worker_init_fn=worker_init_fn,
)

# ---- 5) モデル構築 ----
model = NetAE(
	backbone=cfg.backbone,
	pretrained=True,
	stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
	pre_stages=2,
	pre_stage_strides=((1, 1), (1, 2)),
).to(device)

if 'caformer' in cfg.backbone:
	adjust_first_conv_padding(model.backbone, padding=(3, 3))

# ---- 6) 重みの読み込み（EMA優先, なければ通常）----
if getattr(cfg, 'resume', None):
        ckpt = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        state = ckpt.get('model_ema', ckpt.get('model', ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
                f'[load] missing={len(missing)} unexpected={len(unexpected)} from {cfg.resume}'
        )
else:
        print('[load] cfg.resume が未設定のため、ランダム初期化のまま')

if getattr(cfg, 'model', None) and getattr(cfg.model, 'use_offset_input', False):
        inflate_input_convs_to_2ch(model, verbose=True, init_mode="duplicate")

model.eval()

# ---- 7) 動作確認（任意）：1バッチ取り出してデバイスに載せる ----
if len(valid_dataset) > 0:
        x_masked, x_tgt, mask_or_none, meta = next(iter(val_loader))
        x_masked = x_masked.to(device, non_blocking=True)
        x_in = x_masked
        if getattr(cfg, 'model', None) and getattr(cfg.model, 'use_offset_input', False) and ('offsets' in meta):
                offs_ch = make_offset_channel(x_masked, meta['offsets'])
                x_in = torch.cat([x_masked, offs_ch], dim=1)
        with torch.no_grad():
                _ = model(x_in)
        print('[check] val_loader 取り回し & モデル推論 OK')
else:
	print('[check] valid_dataset が空です')


# ---- 1) Config / device ----
with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='base')

device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')


# ---- 2) フィールドごとに .sgy/.npy を収集（最小版）----
def collect_field_files(list_name: str, data_root: str):
	list_path = Path('configs') / list_name
	with open(list_path) as f:
		fields = [
			ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')
		]
	segy_files, fb_files = [], []
	for field in fields:
		d = Path(data_root) / field
		segy = sorted(list(d.glob('*.sgy')) + list(d.glob('*.segy')))
		fb = sorted(d.glob('*.npy'))
		if not segy or not fb:
			print(f'[warn] skip: {field}')
			continue
		segy_files.append(segy[0])
		fb_files.append(fb[0])
	if not segy_files:
		raise RuntimeError(f'No usable fields in {list_name}')
	return segy_files, fb_files


_, valid_fb_files = collect_field_files(cfg.valid_field_list, cfg.data_root)
valid_segy_files, _ = collect_field_files(cfg.valid_field_list, cfg.data_root)

# ---- 3) 検証用データセット（Augなし・flipなし）----
valid_dataset_full = MaskedSegyGather(
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


# 「サンプル数を固定したい」場合は最初の n アイテムに凍結
class FrozenValDataset(Dataset):
	def __init__(self, items):
		self.items = items

	def __len__(self):
		return len(self.items)

	def __getitem__(self, i):
		return self.items[i]


n_val_item = cfg.val_steps * cfg.batch_size
val_src = copy.copy(valid_dataset_full)  # file_infos を共有
val_src.flip = False
val_items = [val_src[i] for i in range(n_val_item)]
valid_dataset = FrozenValDataset(val_items)

# ---- 4) DataLoader（検証用）----
val_loader = DataLoader(
	valid_dataset,
	batch_size=cfg.batch_size,
	sampler=SequentialSampler(valid_dataset),
	shuffle=False,
	num_workers=0,
	pin_memory=True,
	collate_fn=segy_collate,
	drop_last=False,
	worker_init_fn=worker_init_fn,
)

# ---- 5) モデル構築 ----
model = NetAE(
	backbone=cfg.backbone,
	pretrained=True,
	stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
	pre_stages=2,
	pre_stage_strides=((1, 1), (1, 2)),
).to(device)

if 'caformer' in cfg.backbone:
	adjust_first_conv_padding(model.backbone, padding=(3, 3))

# ---- 6) 重みの読み込み（EMA優先, なければ通常）----
weight = '/workspace/proc/result/fb_seg/train_field_list_wotstkres_edgenext_b36.sail_in22k_ft_in1k_finetune_augspace01/model_180.pth'
ckpt = torch.load(weight, map_location='cpu', weights_only=False)
state = ckpt.get('model_ema', ckpt.get('model', ckpt))
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'[load] missing={len(missing)} unexpected={len(unexpected)} from {cfg.resume}')

if getattr(cfg, 'model', None) and getattr(cfg.model, 'use_offset_input', False):
        inflate_input_convs_to_2ch(model, verbose=True, init_mode="duplicate")

model.eval()

debug_prior_visual(
	model,
	val_loader,
	device,
	cfg.loss.fb_seg,
	outdir='debug_prior',
	num_traces_to_plot=6,
)

# %%

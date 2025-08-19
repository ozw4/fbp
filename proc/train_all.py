# %%
import copy
import datetime
import math
import os
import random
import time
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import segyio
import torch
import torch.nn.functional as F
import utils
from ema import ModelEMA
from hydra import compose, initialize
from loss import shift_robust_l2_pertrace_vec
from metrics import prepare_fb_windows, snr_improvement_from_cached_windows
from model import NetAE
from scipy.ndimage import zoom as nd_zoom
from scipy.signal import resample_poly
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from utils import WarmupCosineScheduler, set_seed
from vis import visualize_pair_quartet, visualize_recon_triplet


class MaskedSegyGather(Dataset):
	def __init__(
		self,
		segy_files: list[str],
		fb_files: list[str],
		ffid_byte=segyio.TraceField.FieldRecord,
		chno_byte=segyio.TraceField.TraceNumber,
		mask_ratio: float = 0.5,
		pick_ratio: float = 0.3,
		target_len: int = 6016,
		flip: bool = False,
		# === 追加: 伸縮オーグメンテーション設定 ===
		augment_time_prob: float = 0.0,  # 学習: 0.3 など / 検証: 0.0
		augment_time_range: tuple[float, float] = (0.95, 1.05),
		augment_space_prob: float = 0.0,  # 学習: 0.3 など / 検証: 0.0
		augment_space_range: tuple[float, float] = (0.90, 1.10),
	):
		self.segy_files = segy_files
		self.fb_files = fb_files
		self.ffid_byte = ffid_byte
		self.chno_byte = chno_byte
		self.mask_ratio = mask_ratio
		self.flip = flip
		self.pick_ratio = pick_ratio
		self.target_len = target_len

		# 追加: オーグメンテーション設定
		self.augment_time_prob = augment_time_prob
		self.augment_time_range = augment_time_range
		self.augment_space_prob = augment_space_prob
		self.augment_space_range = augment_space_range

		# 各ファイルのインデックス等を構築
		self.file_infos = []
		for segy_path, fb_path in zip(self.segy_files, self.fb_files, strict=False):
			print(f'Loading {segy_path} and {fb_path}')
			f = segyio.open(segy_path, 'r', ignore_geometry=True)

			mmap = f.trace.raw[:]
			ffid_values = f.attributes(self.ffid_byte)[:]
			ffid_key_to_indices = self._build_index_map(ffid_values)
			ffid_unique_keys = list(ffid_key_to_indices.keys())
			chno_values = f.attributes(self.chno_byte)[:]
			chno_key_to_indices = self._build_index_map(chno_values)
			chno_unique_keys = list(chno_key_to_indices.keys())
			fb = np.load(fb_path)

			self.file_infos.append(
				dict(
					path=segy_path,
					mmap=mmap,
					ffid_values=ffid_values,
					ffid_key_to_indices=ffid_key_to_indices,
					ffid_unique_keys=ffid_unique_keys,
					chno_values=chno_values,
					chno_key_to_indices=chno_key_to_indices,
					chno_unique_keys=chno_unique_keys,
					n_samples=f.samples.size,
					n_traces=f.tracecount,
					segy_obj=f,
					fb=fb,
				)
			)

	# ---------- 追加: 伸縮ユーティリティ ----------
	def _time_stretch_poly(
		self, x_hw: np.ndarray, factor: float, target_len: int
	) -> np.ndarray:
		"""(H,W) を時間軸だけ factor 倍にストレッチ → W=target_len に再整形"""
		if abs(factor - 1.0) < 1e-4:
			# SciPy なし / 係数ほぼ1.0 → 長さだけ合わせて返す
			return self._fit_time_len_np(x_hw, target_len)

		H, W = x_hw.shape
		frac = Fraction(factor).limit_denominator(128)
		up, down = frac.numerator, frac.denominator
		y = np.stack(
			[resample_poly(x_hw[h], up, down, padtype='line') for h in range(H)], axis=0
		)
		return self._fit_time_len_np(y, target_len)

	def _fit_time_len_np(self, x_hw: np.ndarray, target_len: int) -> np.ndarray:
		"""(H,W') → (H,target_len) に切り出し/パッド（Dataset 内部用の numpy 版）"""
		W = x_hw.shape[1]
		if target_len == W:
			return x_hw
		if target_len < W:
			start = np.random.randint(0, W - target_len + 1)
			return x_hw[:, start : start + target_len]
		# pad
		pad = target_len - W
		return np.pad(x_hw, ((0, 0), (0, pad)), mode='constant')

	def _spatial_stretch_sameH(self, x_hw: np.ndarray, factor: float) -> np.ndarray:
		"""(H,W) をトレース軸だけストレッチしつつ H を元の値に保つ（線形補間）"""
		if abs(factor - 1.0) < 1e-4:
			return x_hw
		H, W = x_hw.shape
		H2 = max(1, int(round(H * factor)))
		# 一旦 H2 にスケール → さらに H に戻す（線形）
		y = nd_zoom(x_hw, zoom=(H2 / H, 1.0), order=1, mode='reflect', prefilter=False)
		y = nd_zoom(y, zoom=(H / H2, 1.0), order=1, mode='reflect', prefilter=False)
		# 万一の off-by-one を補正
		if y.shape[0] < H:
			y = np.pad(y, ((0, H - y.shape[0]), (0, 0)), mode='edge')
		elif y.shape[0] > H:
			y = y[:H, :]
		return y

	# ---------------------------------------------

	def _fit_time_len(
		self, x: np.ndarray, start: int | None = None
	) -> tuple[np.ndarray, int]:
		T, target = x.shape[1], self.target_len
		if start is None:
			start = np.random.randint(0, max(1, T - target + 1)) if target < T else 0
		if target < T:
			return x[:, start : start + target], start
		if target > T:
			pad = target - T
			return np.pad(x, ((0, 0), (0, pad)), mode='constant'), start
		return x, start

	def _build_index_map(self, key_array: np.ndarray) -> dict[int, np.ndarray]:
		uniq, inv, counts = np.unique(
			key_array, return_inverse=True, return_counts=True
		)
		sort_idx = np.argsort(inv, kind='mergesort')
		split_points = np.cumsum(counts)[:-1]
		groups = np.split(sort_idx, split_points)
		return {int(k): g.astype(np.int32) for k, g in zip(uniq, groups, strict=False)}

	def __len__(self):
		return 10**6

	def __getitem__(self, _=None):
		# --- gather 選択 ---
		while True:
			info = random.choice(self.file_infos)
			mmap = info['mmap']
			key_name = random.choice(['ffid', 'chno'])
			unique_keys = info[f'{key_name}_unique_keys']
			key_to_indices = info[f'{key_name}_key_to_indices']

			key = random.choice(unique_keys)
			indices = key_to_indices[key]
			n_total = len(indices)

			if n_total >= 128:
				start_idx = random.randint(0, n_total - 128)
				selected_indices = indices[start_idx : start_idx + 128]
				pad_len = 0
			else:
				selected_indices = indices
				pad_len = 128 - n_total

			fb_subset = info['fb'][selected_indices]
			if pad_len > 0:
				fb_subset = np.concatenate(
					[fb_subset, np.zeros(pad_len, dtype=fb_subset.dtype)]
				)

			pick_ratio = np.count_nonzero(fb_subset > 0) / len(fb_subset)
			if pick_ratio >= self.pick_ratio:
				break

		# --- 読み出し & 正規化 ---
		x = mmap[selected_indices].astype(np.float32, copy=False)  # (H, W_all)
		if pad_len > 0:
			pad_tr = np.zeros((pad_len, x.shape[1]), dtype=np.float32)
			x = np.concatenate([x, pad_tr], axis=0)

		x = x - np.mean(x, axis=1, keepdims=True)
		x = x / (np.std(x, axis=1, keepdims=True) + 1e-10)

		# --- 反転（H軸） ---
		if self.flip and random.random() < 0.5:
			x = np.flip(x, axis=0).copy()
			fb_subset = fb_subset[::-1].copy()

		# --- 伸縮オーグメンテーション（学習のみ確率で） ---
		# 時間ストレッチ（W軸）
		if self.augment_time_prob > 0 and random.random() < self.augment_time_prob:
			f_t = random.uniform(*self.augment_time_range)
			x = self._time_stretch_poly(x, f_t, target_len=self.target_len)
		else:
			# ストレッチしない場合も最終的に target_len へ
			x = self._fit_time_len_np(x, self.target_len)

		# 空間ストレッチ（H軸）
		if self.augment_space_prob > 0 and random.random() < self.augment_space_prob:
			f_h = random.uniform(*self.augment_space_range)
			x = self._spatial_stretch_sameH(x, f_h)

		# --- マスク生成（H軸でトレース置換） ---
		H = x.shape[0]
		num_mask = int(self.mask_ratio * H)
		mask_idx = random.sample(range(H), num_mask) if num_mask > 0 else []
		x_masked = x.copy()
		if num_mask > 0:
			x_masked[mask_idx] = np.random.normal(0.0, 1.0, size=(num_mask, x.shape[1]))

		# ここからは x / xm は同じ開始点で既に target_len に揃っている

		# FB の時間窓座標（クロップ後）を計算：時間軸の start は _time_stretch_poly 内でランダム切り出し済み
		# ここではストレッチしていない前提のときのみ start を考慮（上で _fit_time_len_np を使った場合）
		# → fb_subset は学習には使わないため、学習時の整合性は不要。検証では augment prob=0にしてください。
		fb_idx_win = fb_subset.astype(np.int64)
		# 範囲外や欠損は -1
		invalid = (fb_idx_win <= 0) | (fb_idx_win >= self.target_len)
		fb_idx_win[invalid] = -1

		# Tensor 化（Conv2d想定: (1,H,W)）
		x = torch.from_numpy(x)[None, ...]
		xm = torch.from_numpy(x_masked)[None, ...]
		fb_idx_t = torch.from_numpy(fb_idx_win)

		return {
			'masked': xm,  # (1, H, W)
			'original': x,  # (1, H, W)
			'mask_indices': mask_idx,
			'fb_idx': fb_idx_t,  # (H,)
			'key_name': key_name,
			'indices': selected_indices,
			'file_path': info['path'],
		}


def make_mask_2d(mask_indices_list, H, W, device):
	"""mask_indices_list: 長さB。各要素は「マスクしたtraceインデックス(list[int])」
	出力: (B, 1, H, W) の {0,1} マスク（1=マスク部）
	"""
	B = len(mask_indices_list)
	mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
	for b, idxs in enumerate(mask_indices_list):
		if len(idxs) == 0:
			continue
		mask[b, 0, idxs, :] = 1.0
	return mask


def segy_collate(batch):
	"""batch: list[dict] from MaskedSegyGather.__getitem__
	returns:
		x_masked: (B,1,H,W)
		x_orig:   (B,1,H,W)
		mask_2d:  (B,1,H,W)  # 1=マスク対象トレース
		meta:     dict
	"""
	x_masked = torch.stack([b['masked'] for b in batch], dim=0)  # (B,1,H,W)
	x_orig = torch.stack([b['original'] for b in batch], dim=0)  # (B,1,H,W)
	B, _, H, W = x_masked.shape

	mask_2d = torch.zeros((B, 1, H, W), dtype=x_masked.dtype)
	for i, b in enumerate(batch):
		idxs = b['mask_indices']
		if idxs:
			mask_2d[i, 0, idxs, :] = 1.0
	fb_idx = torch.stack([b['fb_idx'] for b in batch], dim=0)

	meta = {
		'file_path': [b['file_path'] for b in batch],
		'key_name': [b['key_name'] for b in batch],
		'indices': [b['indices'] for b in batch],
		'mask_indices': [b['mask_indices'] for b in batch],
		'fb_idx': fb_idx,
	}
	return x_masked, x_orig, mask_2d, meta


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
):
	global step
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
		x_masked, x_orig, mask_2d, meta = batch
		start_time = time.time()

		x_masked = x_masked.to(device, non_blocking=True)
		x_orig = x_orig.to(device, non_blocking=True)
		mask_2d = mask_2d.to(device, non_blocking=True)

		device_type = (
			'cuda' if torch.cuda.is_available() and 'cuda' in str(device) else 'cpu'
		)
		with autocast(device_type=device_type, enabled=use_amp):
			pred = model(x_masked)

			total_loss = criterion(
				pred, x_orig, mask=mask_2d, max_shift=max_shift, reduction='mean'
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


def criterion(
	pred: torch.Tensor,
	gt: torch.Tensor,
	*,
	mask: torch.Tensor | None = None,  # (B,1,H,W)
	max_shift: int = 8,
	reduction: str = 'mean',
):
	# トレース毎のシフト探索つきL2（ベクトル化版）
	return shift_robust_l2_pertrace_vec(
		pred, gt, mask=mask, max_shift=max_shift, reduction=reduction
	)


@torch.no_grad()
def cover_all_traces_predict(
	model: torch.nn.Module,
	x: torch.Tensor,  # (B,1,H,W) もとの入力（検証用の“ノイジー”）
	*,
	mask_ratio: float = 0.5,
	noise_std: float = 1.0,
	use_amp: bool = True,
	device=None,
	seed: int | None = 12345,  # 再現性用。Noneなら現在のRNGを使う
	passes_batch: int = 4,  # 何パス分まとめて一括推論するか
):
	assert x.dim() == 4 and x.size(1) == 1, 'x must be (B,1,H,W)'
	device = device or x.device
	B, _, H, W = x.shape

	# 1パスでマスクする本数（H-1にクリップして文脈ゼロを回避）
	m = max(1, min(int(round(mask_ratio * H)), H - 1))
	K = math.ceil(H / m)  # 必要パス数（全トレースを覆う）

	# 出力を貯める
	y_full = torch.empty_like(x)

	# 各サンプル独立で処理（サンプルごとに別のマスク分割を作る）
	for b in range(B):
		# 決定論マスクのためのトレース順序
		if seed is not None:
			g = torch.Generator(device='cpu').manual_seed(seed + b)
			perm = torch.randperm(H, generator=g)  # CPUで作ってOK
		else:
			perm = torch.randperm(H)
		chunks = [perm[i : i + m] for i in range(0, H, m)]  # K 個の塊（最後は小さめ可）

		# まとめ推論ループ
		for s in range(0, K, passes_batch):
			batch_chunks = chunks[s : s + passes_batch]
			# まとめ入力を作成
			xmb = []
			for idxs in batch_chunks:
				xm = x[b : b + 1].clone()  # (1,1,H,W)
				# ノイズはCPUで決定論的に生成→GPUへ
				if seed is not None:
					gk = torch.Generator(device='cpu').manual_seed(
						(seed + b) * 100003 + s * 1009 + int(idxs[0])
					)
					n = (
						torch.randn((1, 1, len(idxs), W), generator=gk, device='cpu')
						* noise_std
					)
				else:
					n = torch.randn((1, 1, len(idxs), W), device='cpu') * noise_std
				n = n.to(device=device, non_blocking=True)
				idxs_dev = idxs.to(device)
				xm[:, :, idxs_dev, :] = n  # トレース方向に置換
				xmb.append(xm)
			xmb = torch.cat(xmb, dim=0)  # (len(batch_chunks),1,H,W)

			# 推論
			dev_type = 'cuda' if xmb.is_cuda else 'cpu'
			with autocast(device_type=dev_type, enabled=use_amp):
				yb = model(xmb)  # (len(batch_chunks),1,H,W)

			# 合成：各パスでマスクしていたトレースを採用
			for k, idxs in enumerate(batch_chunks):
				y_full[b, :, idxs.to(device), :] = yb[k, :, idxs.to(device), :]

	return y_full


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
			noise_std=1.0,
			use_amp=True,
			device=device,
			seed=cfg_snr.seed,
			passes_batch=cfg_snr.passes_batch,
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

		# 可視化（任意）
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


@torch.no_grad()
def cover_all_traces_predict_chunked(
	model,
	x,  # (B,1,H,W)
	*,
	chunk_h: int = 128,  # 1タイルのトレース本数
	overlap: int = 32,  # タイル間の重なり
	mask_ratio: float = 0.5,
	noise_std: float = 1.0,
	use_amp: bool = True,
	device=None,
	seed: int = 12345,
	passes_batch: int = 4,
):
	"""H軸(トレース)をタイル分割し、各タイルで cover_all_traces_predict を実行。
	オーバーラップ領域は線形ランプで重み付け平均して縫い合わせる。
	"""
	assert overlap < chunk_h, 'overlap は chunk_h より小さくしてください'
	device = device or x.device
	B, _, H, W = x.shape

	y_acc = torch.zeros_like(x)
	w_acc = torch.zeros((B, 1, H, 1), dtype=x.dtype, device=device)
	step = chunk_h - overlap

	s = 0
	while s < H:
		e = min(s + chunk_h, H)
		xt = x[:, :, s:e, :]  # (B,1,h_t,W)

		# タイルごとに全トレースカバー再構成
		yt = cover_all_traces_predict(
			model,
			xt,
			mask_ratio=mask_ratio,
			noise_std=noise_std,
			use_amp=use_amp,
			device=device,
			seed=seed + s,  # タイル毎にシード変化
			passes_batch=passes_batch,
		)  # (B,1,h_t,W)

		# ── 重み（H方向のランプ）を作成 ──
		h_t = e - s
		w = torch.ones((1, 1, h_t, 1), dtype=x.dtype, device=device)
		left_ov = min(overlap, s)  # 先頭側オーバーラップ量
		right_ov = min(overlap, H - e)  # 末尾側オーバーラップ量

		if left_ov > 0:
			ramp = torch.linspace(
				0, 1, steps=left_ov, device=device, dtype=x.dtype
			).view(1, 1, -1, 1)
			w[:, :, :left_ov, :] = ramp
		if right_ov > 0:
			ramp = torch.linspace(
				1, 0, steps=right_ov, device=device, dtype=x.dtype
			).view(1, 1, -1, 1)
			w[:, :, -right_ov:, :] = ramp

		# 蓄積
		y_acc[:, :, s:e, :] += yt * w
		w_acc[:, :, s:e, :] += w

		if e == H:
			break
		s += step

	y_full = y_acc / (w_acc + 1e-8)
	return y_full


def _read_gather_by_indices(
	f: segyio.SegyFile,
	indices: np.ndarray,
	*,
	target_len: int,
) -> np.ndarray:
	"""個々のトレースを読み出して (H, target_len) に切り揃える（堅牢版）"""
	traces = []
	for i in np.sort(indices):
		a = np.asarray(f.trace[i], dtype=np.float32)  # 可変長でもOK
		if a.shape[0] >= target_len:
			a = a[:target_len]
		else:
			pad = target_len - a.shape[0]
			a = np.pad(a, (0, pad), mode='constant')
		traces.append(a)
	if len(traces) == 0:
		return np.empty((0, target_len), dtype=np.float32)
	return np.stack(traces, axis=0)  # (H, target_len)


# 1ショット（noisy/clean）を読み込んで (Nshots,1,Hmax,W) へ
def load_synth_pair(
	noisy_path: str,
	clean_path: str,
	*,
	extract_key1idxs: Sequence[int] | int = (401,),
	target_len: int = 6016,
	standardize: bool = True,
	endian: str = 'little',
):
	"""指定した FFID の gather だけ抽出して返す。
	返り値:
	  x_noisy:    (N,1,Hmax,W)
	  x_clean:    (N,1,Hmax,W)
	  valid_mask: (N,1,Hmax,1)  1=実トレース, 0=パディング
	  used_key1idxs: list[int]     実際に抽出できたFFID
	  H_list:     list[int]     各ショットの元のトレース本数
	"""
	if isinstance(extract_key1idxs, int):
		extract_key1idxs = [extract_key1idxs]

	used_ffids = []
	gathers_noisy = []
	gathers_clean = []
	H_list = []

	# まず属性だけ読んで FFID→indices を作る
	with (
		segyio.open(
			noisy_path, 'r', ignore_geometry=True, strict=False, endian=endian
		) as fn,
		segyio.open(
			clean_path, 'r', ignore_geometry=True, strict=False, endian=endian
		) as fc,
	):
		cdp_tr = np.asarray(fn.attributes(segyio.TraceField.CDP_TRACE)[:])

		for ff in extract_key1idxs:
			idx = np.where(cdp_tr == ff)[0]

			# 個々のトレースを読み出し→ target_len に揃える
			g_noisy = _read_gather_by_indices(fn, idx, target_len=target_len)  # (H,W)
			g_clean = _read_gather_by_indices(fc, idx, target_len=target_len)  # (H,W)

			# per-trace 標準化（学習と合わせる）
			if standardize and g_noisy.shape[0] > 0:
				for x in (g_noisy, g_clean):
					m = x.mean(axis=1, keepdims=True)
					s = x.std(axis=1, keepdims=True) + 1e-10
					x -= m
					x /= s

			gathers_noisy.append(g_noisy)
			gathers_clean.append(g_clean)
			used_ffids.append(int(ff))
			H_list.append(int(g_noisy.shape[0]))

	if len(gathers_noisy) == 0:
		raise ValueError('指定した FFID がいずれのファイルにも見つかりませんでした。')

	# H をバッチ内の最大にパディングして (N,1,Hmax,W) に揃える
	N = len(gathers_noisy)
	Hmax = max(H_list)
	W = target_len
	x_noisy = torch.zeros((N, 1, Hmax, W), dtype=torch.float32)
	x_clean = torch.zeros((N, 1, Hmax, W), dtype=torch.float32)
	valid_mask = torch.zeros((N, 1, Hmax, 1), dtype=torch.float32)

	for i, (gn, gc, H) in enumerate(
		zip(gathers_noisy, gathers_clean, H_list, strict=False)
	):
		x_noisy[i, 0, :H, :] = torch.from_numpy(gn)
		x_clean[i, 0, :H, :] = torch.from_numpy(gc)
		valid_mask[i, 0, :H, 0] = 1.0

	return x_noisy, x_clean, valid_mask, used_ffids, H_list


@torch.no_grad()
def eval_synthe(
	x_clean,
	pred,
	device=None,
):
	mses, maes, psnrs = [], [], []
	for p, gt in zip(pred, x_clean, strict=False):
		if device is not None:
			p, gt = p.to(device), gt.to(device)
		mse = F.mse_loss(p, gt).item()
		mae = F.l1_loss(p, gt).item()
		psnr = -10.0 * torch.log10(F.mse_loss(p, gt)).item()  # 標準化前提

		mses.append(mse)
		maes.append(mae)
		psnrs.append(psnr)

	out = {
		'mse': float(sum(mses) / len(mses)),
		'mae': float(sum(maes) / len(maes)),
		'psnr': float(sum(psnrs) / len(psnrs)),
		'num_shots': len(x_clean),
	}
	return out


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
)
valid_dataset = MaskedSegyGather(
	segy_files,
	fb_files,
	mask_ratio=0.5,
	flip=False,
	augment_time_prob=0.0,
	augment_space_prob=0.0,
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
	backbone=cfg.backbone,
	pretrained=True,
).to(device)

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

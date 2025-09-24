# %%
# proc/test.inference_tta.py
"""Collect ALL logits per view in raw trace order, evaluate per view,
then cross-merge by PoE (sum of log-softmax) or logits_sum (configurable).

- Views: horizontal flip only -> {none, hflip}
- Domains: shot(=ffid), recv(=chno), cmp(=cmp)
- Loop order: for domain in domains: for view in [none, hflip]
- For each (domain, view):
    * run model WITHOUT internal TTA
    * invert flip so logits return to the **original trace order**
    * reconstruct a single (N_total_traces, W) array by averaging overlapped windows **in raw global order**
    * evaluate Hit@{0,2,4,8} ms on this view
    * (optionally) save that 2D array to disk: <out_root>/<domain>_<view>.npy
- After all views are processed:
    * build all_logits = [shot_none, shot_hflip, recv_none, recv_hflip, cmp_none, cmp_hflip]
    * do **cross-domain merge** in raw-global order
        - infer.cross_merge_method: poe | logits_sum  (default: poe)
        - infer.cross_view_weights: {shot:1.0, recv:1.0, cmp:1.0, hflip:1.0, "shot:hflip":0.95, ...}
        - infer.cross_view_temps:   {shot_none:1.0, recv:1.0, ...}  # PoE用（任意）

Usage (例)
---------
python -m proc.test.inference_tta \
  infer.ckpt=/path/to/ckpt.pt \
  infer.tta_views="[none,hflip]" \
  infer.domains="[shot,recv,cmp]" \
  infer.cross_merge_method=poe \
  infer.cross_view_weights='{shot:1.0, recv:0.8, cmp:0.9, hflip:0.97}'
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm

from proc.util.dataset import MaskedSegyGather
from proc.util.features import (
	make_offset_channel_phys,
	make_time_channel,
)
from proc.util.model import NetAE, adjust_first_conv_padding
from proc.util.model_utils import inflate_input_convs_to_2ch, inflate_input_convs_to_nch
from proc.util.utils import collect_field_files, set_seed

# -------------------------
# Model helpers
# -------------------------


def build_model_from_cfg(cfg) -> torch.nn.Module:
	model = NetAE(
		backbone=cfg.backbone,
		pretrained=True,
		stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
		pre_stages=2,
		pre_stage_strides=((1, 1), (1, 2)),
	)
	if getattr(cfg.model, 'first_conv_same_pad', False):
		adjust_first_conv_padding(model.backbone, padding=(1, 1))
	if getattr(cfg.model, 'use_offset_input', False) and getattr(
		cfg.model, 'use_time_input', True
	):
		inflate_input_convs_to_nch(
			model if 'model_without_ddp' in locals() else model,
			3,
			verbose=True,
			init_mode='duplicate',
		)
	elif getattr(cfg.model, 'use_offset_input', False):
		inflate_input_convs_to_2ch(
			model if 'model_without_ddp' in locals() else model,
			verbose=True,
			init_mode='duplicate',
		)
	return model


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
	state = torch.load(str(path), map_location='cpu', weights_only=False)
	for k in ['model_ema', 'state_dict', 'model']:
		if k in state and isinstance(state[k], dict):
			model.load_state_dict(state[k], strict=False)
			print(f"[ckpt] loaded via key='{k}'")
			return
	model.load_state_dict(state, strict=False)
	print('[ckpt] loaded raw dict')


# -------------------------
# Views
# -------------------------

TTAName = Literal['none', 'hflip']


def _hflip(x: torch.Tensor) -> torch.Tensor:
	return x.flip(dims=(2,))  # traces axis (H)


def apply_view(x: torch.Tensor, view: TTAName) -> torch.Tensor:
	return x if view == 'none' else _hflip(x)


def invert_view(x: torch.Tensor, view: TTAName) -> torch.Tensor:
	return x if view == 'none' else _hflip(x)


# -------------------------
# Deterministic inference enumerator (windows) with raw-global index
# -------------------------


class InferenceGatherWindows(MaskedSegyGather):
	"""Deterministic window enumerator for inference (no random aug).
	Additionally provides:
	  - raw_idx: (Hwin,) raw-global row index for each trace in the window
	  - n_total: scalar, total # of traces across all files (same for all batches)
	"""

	def __init__(
		self,
		segy_files,
		fb_files,
		*,
		domain: Literal['shot', 'recv', 'cmp'],
		win: int = 128,
		stride: int = 64,
		pad_last: bool = True,
		target_len: int = 6016,
		**super_kwargs,
	):
		super().__init__(
			segy_files,
			fb_files,
			primary_keys=None,
			use_superwindow=False,
			mask_ratio=0.0,
			pick_ratio=1.0,
			flip=False,
			augment_time_prob=0.0,
			augment_space_prob=0.0,
			augment_freq_prob=0.0,
			target_mode='fb_seg',
			target_len=int(target_len),
			**super_kwargs,
		)
		self.domain = domain
		self.win = int(win)
		self.stride = int(stride)
		self.pad_last = bool(pad_last)
		self.items: list[
			tuple[int, int, int, int, int]
		] = []  # (file_idx, pk, s, e, Htot)

		# raw-global base index per file
		self._file_base = []
		base = 0
		for info in self.file_infos:
			n_tr = len(info['ffid_values'])
			self._file_base.append(base)
			base += n_tr
		self.n_total = base  # total # of traces over all files

		# enumerate windows per primary-key in each domain
		for fi, info in enumerate(self.file_infos):
			if domain == 'shot':
				k2i = info.get('ffid_key_to_indices')
				prim_all = info.get('ffid_values')
				sec_all = info.get('chno_values')
			elif domain == 'recv':
				k2i = info.get('chno_key_to_indices')
				prim_all = info.get('chno_values')
				sec_all = info.get('ffid_values')
			elif domain == 'cmp':
				k2i = info.get('cmp_key_to_indices')
				prim_all = info.get('cmp_values')
				sec_all = info.get('offsets')
				if k2i is None:
					continue
			else:
				raise ValueError(f'unsupported domain: {domain}')
			if not k2i:
				continue

			for pk, idxs in sorted(k2i.items()):
				prim = prim_all[idxs]
				sec = sec_all[idxs]
				o = np.argsort(prim, kind='mergesort')
				idxs = idxs[o]
				sec = sec[o]
				o2 = np.argsort(sec, kind='mergesort')
				idxs = idxs[o2]
				H = len(idxs)
				if H <= 0:
					continue

				if self.win >= H:
					starts = [0]
				else:
					starts = list(range(0, H - self.win + 1, self.stride))
					if self.pad_last and (starts[-1] + self.win < H):
						starts.append(H - self.win)

				for s in starts:
					e = min(s + self.win, H)
					self.items.append((fi, int(pk), int(s), int(e), H))

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, i: int):
		fi, pk, s, e, Htot = self.items[i]
		info = self.file_infos[fi]

		if self.domain == 'shot':
			idxs = info['ffid_key_to_indices'][pk]
			prim_vals = info['ffid_values'][idxs]
			sec_vals = info['chno_values'][idxs]
		elif self.domain == 'recv':
			idxs = info['chno_key_to_indices'][pk]
			prim_vals = info['chno_values'][idxs]
			sec_vals = info['ffid_values'][idxs]
		else:  # cmp
			idxs = info['cmp_key_to_indices'][pk]
			prim_vals = info['cmp_values'][idxs]
			sec_vals = info['offsets'][idxs]

		o = np.argsort(prim_vals, kind='mergesort')
		idxs = idxs[o]
		sec_vals = sec_vals[o]
		o2 = np.argsort(sec_vals, kind='mergesort')
		idxs = idxs[o2]

		idx_win = idxs[s:e]
		pad_len = max(0, self.win - (e - s))
		if pad_len > 0 and self.pad_last:
			fill = (
				idx_win[-1] if len(idx_win) > 0 else (idxs[0] if len(idxs) > 0 else 0)
			)
			idx_win = np.concatenate(
				[idx_win, np.full((pad_len,), fill, dtype=idx_win.dtype)]
			)
		idx_win = np.asarray(idx_win, dtype=np.int64)

		# inputs
		x = info['mmap'][idx_win].astype(np.float32)
		x = x - x.mean(axis=1, keepdims=True)
		x = x / (x.std(axis=1, keepdims=True) + 1e-10)

		x, _ = self._fit_time_len(x, start=0)
		W = x.shape[1]

		# labels
		fb = info['fb'][idx_win].astype(np.int64)
		fb_idx_win = fb.copy()
		fb_idx_win[(fb_idx_win <= 0) | (fb_idx_win >= W)] = -1

		abs_h = np.arange(s, s + len(idx_win), dtype=np.int64)
		if pad_len > 0 and self.pad_last:
			abs_h[-pad_len:] = -1
			fb_idx_win[abs_h == -1] = -1

		offsets = info['offsets'][idx_win].astype(np.float32)
		ffid_vals = info['ffid_values'][idx_win].astype(np.int64)
		chno_vals = info['chno_values'][idx_win].astype(np.int64)
		cmp_vals = (
			info['cmp_values'][idx_win].astype(np.int64)
			if info.get('cmp_values') is not None
			else np.zeros(len(idx_win), dtype=np.int64)
		)

		# raw-global row index for this window
		raw_idx_global = self._file_base[fi] + idx_win  # (Hwin,)

		return {
			'x': torch.from_numpy(x)[None, ...],
			'fb_idx': torch.from_numpy(fb_idx_win),
			'offsets': torch.from_numpy(offsets),
			'dt_sec': torch.tensor(info['dt_sec'], dtype=torch.float32),
			'ffid': torch.from_numpy(ffid_vals),
			'chno': torch.from_numpy(chno_vals),
			'cmp': torch.from_numpy(cmp_vals),
			'group_id': f'{fi}:{self.domain}:{pk}',
			'abs_h': torch.from_numpy(abs_h),
			'gather_len': torch.tensor(Htot, dtype=torch.int64),
			'domain': self.domain,
			'raw_idx': torch.from_numpy(raw_idx_global),
			'n_total': torch.tensor(self.n_total, dtype=torch.int64),
		}


def make_valid_loader_for_domain(
	cfg,
	*,
	domain: Literal['shot', 'recv', 'cmp'],
	batch_size: int | None = None,
	num_workers: int | None = None,
) -> DataLoader:
	data_root = Path(cfg.data_root)
	vlist_name = getattr(cfg, 'valid_field_list', None) or getattr(
		cfg.dataset, 'valid_field_list', None
	)
	if vlist_name is None:
		raise RuntimeError('valid_field_list is not set.')
	valid_field_list = Path(__file__).parent / 'configs' / str(vlist_name)
	segy_files, fb_files = collect_field_files(valid_field_list, data_root)

	ds = InferenceGatherWindows(
		segy_files,
		fb_files,
		domain=domain,
		win=int(getattr(cfg.infer, 'win_size_traces', 128)),
		stride=int(getattr(cfg.infer, 'win_stride_traces', 64)),
		pad_last=bool(getattr(cfg.infer, 'win_pad_last', True)),
		target_len=int(getattr(cfg.dataset, 'target_len', 6016)),
		use_header_cache=getattr(cfg.dataset, 'use_header_cache', False),
		header_cache_dir=getattr(cfg.dataset, 'header_cache_dir', None),
	)
	bs = int(
		batch_size or getattr(cfg.infer, 'batch_size', getattr(cfg, 'batch_size', 1))
	)
	nw = int(
		num_workers or getattr(cfg.infer, 'num_workers', getattr(cfg, 'num_workers', 4))
	)

	def _collate(batch):
		b0 = batch[0]
		x = torch.stack([b['x'] for b in batch])
		meta: dict = {}
		for k in [
			'fb_idx',
			'offsets',
			'dt_sec',
			'ffid',
			'chno',
			'cmp',
			'abs_h',
			'gather_len',
			'raw_idx',
		]:
			meta[k] = torch.stack([b[k] for b in batch])
		meta['group_id'] = [b['group_id'] for b in batch]
		meta['domain'] = b0['domain']
		# n_total is identical for all items -> take first
		meta['n_total'] = batch[0]['n_total']
		return x, None, None, meta

	return DataLoader(
		ds,
		batch_size=bs,
		sampler=SequentialSampler(ds),
		num_workers=nw,
		pin_memory=True,
		drop_last=False,
		collate_fn=_collate,
	)


# -------------------------
# Per-view accumulator (raw-global (N_total, W))
# -------------------------


class ViewAccumulator:
	"""Accumulate logits for one (domain, view) directly in raw-global order, then evaluate."""

	def __init__(self):
		self.sum_: np.ndarray | None = None  # (N_total, W)
		self.cnt_: np.ndarray | None = None  # (N_total,)
		self.fb_: np.ndarray | None = None  # (N_total,)
		self.dt_: np.ndarray | None = None  # (N_total,)
		self.W: int | None = None
		self.N: int | None = None

	def add_batch(self, meta: dict, logits: torch.Tensor, *, view_name: str):
		"""logits: (B,1,Hwin,W) already inverted to original order for this view."""
		arr = logits.to(torch.float32).detach().cpu().numpy()
		B, _, Hwin, W = arr.shape

		raw_idx = meta['raw_idx'].numpy()  # (B,Hwin) raw-global row
		fb_idx = meta['fb_idx'].numpy()  # (B,Hwin)
		dt_sec = meta['dt_sec'].numpy().reshape(-1)  # (B,)
		N_total = int(meta['n_total'].item())

		if self.sum_ is None:
			self.N = N_total
			self.W = W
			self.sum_ = np.zeros((self.N, self.W), dtype=np.float64)
			self.cnt_ = np.zeros((self.N,), dtype=np.int32)
			self.fb_ = np.full((self.N,), -1, dtype=np.int64)
			self.dt_ = np.full((self.N,), np.nan, dtype=np.float32)

		for b in range(B):
			dt_b = float(dt_sec[b])
			for h in range(Hwin):
				r = int(raw_idx[b, h])
				if r < 0:  # padded window tail
					continue
				self.sum_[r, :] += arr[b, 0, h, :]
				self.cnt_[r] += 1
				self.fb_[r] = int(fb_idx[b, h])
				self.dt_[r] = dt_b

	def finalize_logits(self) -> np.ndarray:
		avg = self.sum_.copy()
		nz = self.cnt_ > 0
		if nz.any():
			avg[nz, :] = (avg[nz, :].T / self.cnt_[nz]).T
		return avg.astype(np.float32)

	def finalize_labels(self) -> tuple[np.ndarray, np.ndarray]:
		return self.fb_.copy(), self.dt_.copy()


def eval_view(
	logits_2d: np.ndarray,
	fb_idx: np.ndarray,
	dt_sec: np.ndarray,
	thr_ms: Sequence[float],
) -> dict:
	"""Evaluate Hit@{thr_ms} for one view. logits_2d: (N, W)."""
	if logits_2d.size == 0:
		return {f'hit@{int(t)}': 0.0 for t in thr_ms} | {'n_tr_valid': 0}
	N, W = logits_2d.shape
	pred_idx = logits_2d.argmax(axis=1).astype(
		np.int64
	)  # argmax(logit)==argmax(sigmoid)
	valid = (fb_idx > 0) & (fb_idx < W) & np.isfinite(dt_sec) & (dt_sec > 0)
	diff = np.abs(pred_idx - fb_idx)
	n_valid = int(valid.sum())

	out = {}
	for t in thr_ms:
		tol = np.rint((float(t) / 1000.0) / dt_sec).astype(np.int64)
		hit = (diff <= tol) & valid
		out[f'hit@{int(t)}'] = float(hit.sum()) / max(n_valid, 1)
	out['n_tr_valid'] = n_valid
	return out


# -------------------------
# Cross-domain: logits_sum & PoE (log-softmax sum)
# -------------------------


def _idx_of(domain: str, view: str) -> int:
	mp = {
		('shot', 'none'): 0,
		('shot', 'hflip'): 1,
		('recv', 'none'): 2,
		('recv', 'hflip'): 3,
		('cmp', 'none'): 4,
		('cmp', 'hflip'): 5,
	}
	return mp[(domain, view)]


def _weight(view_weights: dict[str, float] | None, d: str, v: str) -> float:
	# 優先順位: 'domain:view' → 'domain' → 'view' → 1.0
	if view_weights is None:
		return 1.0
	k1 = f'{d}:{v}'
	if k1 in view_weights:
		return float(view_weights[k1])
	if d in view_weights:
		return float(view_weights[d])
	if v in view_weights:
		return float(view_weights[v])
	return 1.0


def _poe_logp(arr: np.ndarray, T: float = 1.0) -> np.ndarray:
	"""arr: (N, W) logits → log_softmax(arr / T, axis=1)"""
	T = max(1e-6, float(T))
	a = arr / T
	a = a - a.max(axis=1, keepdims=True)  # for stability
	logsumexp = np.log(np.exp(a).sum(axis=1, keepdims=True))
	return a - logsumexp


def eval_cross_logits_sum(
	all_logits: list[np.ndarray],
	fb_full: np.ndarray,  # (N_total,) raw order
	dt_full: np.ndarray,  # (N_total,)
	thr_ms: Sequence[float],
	*,
	use: Sequence[tuple[str, str]] = (
		('shot', 'none'),
		('shot', 'hflip'),
		('recv', 'none'),
		('recv', 'hflip'),
		('cmp', 'none'),
		('cmp', 'hflip'),
	),
	view_weights: dict[str, float] | None = None,
) -> dict:
	sel = []
	for d, v in use:
		arr = all_logits[_idx_of(d, v)]
		if arr is None or arr.size == 0:
			continue
		sel.append(((d, v), arr))
	if len(sel) == 0:
		return {f'hit@{int(t)}': 0.0 for t in thr_ms} | {'n_tr_valid': 0}

	N, W = sel[0][1].shape
	total = np.zeros((N, W), dtype=np.float64)
	for (d, v), arr in sel:
		if arr.shape != (N, W):
			raise RuntimeError(f'shape mismatch for {(d, v)}: {arr.shape} vs {(N, W)}')
		total += _weight(view_weights, d, v) * arr.astype(np.float64, copy=False)

	pred_idx = total.argmax(axis=1).astype(np.int64)
	valid = (fb_full > 0) & (fb_full < W) & np.isfinite(dt_full) & (dt_full > 0)
	diff = np.abs(pred_idx - fb_full)
	n_valid = int(valid.sum())

	out = {}
	for t in thr_ms:
		tol = np.rint((float(t) / 1000.0) / dt_full).astype(np.int64)
		hit = (diff <= tol) & valid
		out[f'hit@{int(t)}'] = float(hit.sum()) / max(n_valid, 1)
	out['n_tr_valid'] = n_valid
	return out


def eval_cross_poe(
	all_logits: list[np.ndarray],
	fb_full: np.ndarray,  # (N_total,) raw order
	dt_full: np.ndarray,  # (N_total,)
	thr_ms: Sequence[float],
	*,
	use: Sequence[tuple[str, str]] = (
		('shot', 'none'),
		('shot', 'hflip'),
		('recv', 'none'),
		('recv', 'hflip'),
		('cmp', 'none'),
		('cmp', 'hflip'),
	),
	view_weights: dict[str, float] | None = None,
	view_temps: dict[str, float]
	| None = None,  # key例: 'shot_none', 'recv', 'hflip', 'shot:hflip'
) -> dict:
	sel = []
	names = []
	for d, v in use:
		arr = all_logits[_idx_of(d, v)]
		if arr is None or arr.size == 0:
			continue
		sel.append(arr)
		names.append(f'{d}_{v}')
	if len(sel) == 0:
		return {f'hit@{int(t)}': 0.0 for t in thr_ms} | {'n_tr_valid': 0}

	N, W = sel[0].shape
	total_logp = np.zeros((N, W), dtype=np.float64)

	def _temp_for(name: str) -> float:
		if not view_temps:
			return 1.0
		# 優先順位: exact name → domain → view → default
		if name in view_temps:
			return float(view_temps[name])
		d, v = name.split('_', 1)
		if d in view_temps:
			return float(view_temps[d])
		if v in view_temps:
			return float(view_temps[v])
		return 1.0

	def _weight_for(name: str) -> float:
		if not view_weights:
			return 1.0
		d, v = name.split('_', 1)
		return _weight(view_weights, d, v)

	for name, arr in zip(names, sel, strict=False):
		logp = _poe_logp(arr, T=_temp_for(name))
		total_logp += _weight_for(name) * logp

	pred_idx = total_logp.argmax(axis=1).astype(np.int64)
	valid = (fb_full > 0) & (fb_full < W) & np.isfinite(dt_full) & (dt_full > 0)
	diff = np.abs(pred_idx - fb_full)
	n_valid = int(valid.sum())

	out = {}
	for t in thr_ms:
		tol = np.rint((float(t) / 1000.0) / dt_full).astype(np.int64)
		hit = (diff <= tol) & valid
		out[f'hit@{int(t)}'] = float(hit.sum()) / max(n_valid, 1)
	out['n_tr_valid'] = n_valid
	return out


# -------------------------
# Hydra main
# -------------------------

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
with initialize(config_path='configs', version_base='1.3'):
	cfg = compose(config_name='inference')

print('[cfg]\n' + OmegaConf.to_yaml(cfg))
set_seed(int(getattr(cfg, 'seed', 42)))

device = torch.device(
	str(getattr(cfg, 'device', 'cuda')) if torch.cuda.is_available() else 'cpu'
)
use_amp = bool(getattr(cfg, 'use_amp', True))

model = build_model_from_cfg(cfg).to(device)
ckpt = getattr(cfg.infer, 'ckpt', None)
if ckpt is None:
	raise RuntimeError('Please set infer.ckpt to a checkpoint path')
print(f'[ckpt] loading {ckpt} …')
load_checkpoint(model, ckpt)
model.eval()
# disable any internal TTA in the model if it exists
if hasattr(model, 'use_tta'):
	model.use_tta = False

# config
domains = list(getattr(cfg.infer, 'domains', ['shot', 'recv', 'cmp']))
views_cfg: tuple[TTAName, ...] = tuple(
	getattr(cfg.infer, 'tta_views', ['none', 'hflip'])
)
allowed = {'none', 'hflip'}
if any(v not in allowed for v in views_cfg):
	raise RuntimeError(f'infer.tta_views must be subset of {allowed}')
progress = bool(getattr(cfg.infer, 'progress', True))

# thresholds
fb_cfg = getattr(cfg.loss, 'fb_seg', None)
if fb_cfg is None:
	raise RuntimeError('cfg.loss.fb_seg not found')
thr_ms = tuple(
	float(t) for t in getattr(fb_cfg, 'hit_thresholds_ms', (0.0, 2.0, 4.0, 8.0))
)

# cross-merge method/params
cross_method = str(
	getattr(cfg.infer, 'cross_merge_method', 'poe')
).lower()  # 'poe' | 'logits_sum'
cross_weights = dict(getattr(cfg.infer, 'cross_view_weights', {}))  # optional
cross_temps = dict(getattr(cfg.infer, 'cross_view_temps', {}))  # optional (PoE)

# save option
save_logits = bool(getattr(cfg.infer, 'save_logits', False))
out_root_cfg = getattr(cfg.infer, 'logits_out_root', None)
if save_logits:
	out_root_cfg = (
		Path(out_root_cfg or './_tta_logits') / Path(str(ckpt)).with_suffix('').name
	)
	out_root_cfg.mkdir(parents=True, exist_ok=True)

# containers to return
view_name_list = [
	('shot', 'none'),
	('shot', 'hflip'),
	('recv', 'none'),
	('recv', 'hflip'),
	('cmp', 'none'),
	('cmp', 'hflip'),
]
all_logits: list[np.ndarray] = [np.zeros((0, 0), dtype=np.float32) for _ in range(6)]
per_view_report: dict[str, dict] = {}

# For cross-merge, keep fb/dt from the first finished view (they are raw-global aligned)
fb_full: np.ndarray | None = None
dt_full: np.ndarray | None = None


# -------------------------
# Main: per-view runner
# -------------------------


def run_one_view(
	domain: str, view: TTAName
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
	loader = make_valid_loader_for_domain(cfg, domain=domain)
	acc = ViewAccumulator()
	total_batches = len(loader)
	pbar = tqdm(
		total=total_batches,
		desc=f'[{domain}:{view}] infer',
		unit='batch',
		leave=False,
		disable=not progress,
	)

	# incremental stats for pbar (optional)
	inc_hits = {f'hit@{int(t)}': 0 for t in thr_ms}
	inc_valid = 0

	cfg_obj = cfg if cfg is not None else getattr(model, 'cfg', None)
	use_offset = bool(
		getattr(getattr(cfg_obj, 'model', None), 'use_offset_input', False)
	)
	use_time = bool(getattr(getattr(cfg_obj, 'model', None), 'use_time_input', False))

	model.eval()
	for x, _t, _m, meta in loader:
		x = x.to(device, non_blocking=True)
		# build input (offset channel if requested)
		x_in = x
		if (use_offset or use_time) and ('offsets' in meta):
			offs_ch = make_offset_channel_phys(
				x_like=x,
				offsets_m=meta['offsets'],  # (B,H)
				x95_m=cfg_obj.norm.x95_m,
				mode=getattr(cfg_obj.norm, 'offset_mode', 'log1p'),
				clip_hi=getattr(cfg_obj.norm, 'offset_clip_hi', 1.5),
			).to(device=x.device, dtype=x.dtype)

			if use_time and ('dt_sec' in meta):
				time_ch = make_time_channel(
					x_like=x,
					dt_sec=meta['dt_sec'],  # scalar or (B,)
					t95_ms=cfg_obj.norm.t95_ms,
					clip_hi=getattr(cfg_obj.norm, 'time_clip_hi', 1.5),
				).to(device=x.device, dtype=x.dtype)
				x_in = torch.cat([x, offs_ch, time_ch], dim=1)
		# run single view
		dev_type = 'cuda' if x.is_cuda else 'cpu'
		with autocast(device_type=dev_type, enabled=use_amp):
			xv = apply_view(x_in, view)
			logit = model(xv)  # (B,1,H,W)
			logit = invert_view(logit, view)  # back to original trace order

		# accumulate logits in raw-global order
		acc.add_batch(meta, logit, view_name=view)

		# lightweight per-batch incremental metric for pbar (optional)
		with torch.no_grad():
			prob = torch.sigmoid(logit[:, 0])  # (B,H,W)
			pidx = prob.argmax(dim=-1)  # (B,H)
			fb_idx = meta['fb_idx']
			W = prob.shape[-1]
			valid = (fb_idx > 0) & (fb_idx < W)
			# per-trace tol in samples
			tol_samples = {
				int(t): torch.round((t / 1000.0) / meta['dt_sec'].view(-1, 1)).to(
					torch.long
				)
				for t in thr_ms
			}
			diff = (pidx.cpu() - fb_idx).abs()
			for t in thr_ms:
				hit = (diff <= tol_samples[int(t)]) & valid
				inc_hits[f'hit@{int(t)}'] += int(hit.sum().item())
			inc_valid += int(valid.sum().item())

		if inc_valid > 0:
			h8 = inc_hits.get('hit@8', 0) / max(inc_valid, 1)
			pbar.set_postfix(h8=f'{h8:.3f}', n=inc_valid)
		pbar.update(1)

	pbar.close()

	# finalize & evaluate once
	logits_2d = acc.finalize_logits()  # (N, W)
	fb_idx, dt_sec = acc.finalize_labels()
	report = eval_view(logits_2d, fb_idx, dt_sec, thr_ms)
	return logits_2d, report, fb_idx, dt_sec


# run per view
for dom, vw in view_name_list:
	if dom not in domains or vw not in views_cfg:
		continue
	logits_2d, rep, fb_v, dt_v = run_one_view(dom, vw)
	idx = _idx_of(dom, vw)
	all_logits[idx] = logits_2d
	per_view_report[f'{dom}_{vw}'] = rep
	# save fb/dt once (raw-global; same across views)
	if fb_full is None and logits_2d.size > 0:
		fb_full = fb_v
		dt_full = dt_v

	# save only the 2D logits per view (no UID-splitting)
	if save_logits:
		out_path = Path(out_root_cfg) / f'{dom}_{vw}.npy'
		np.save(out_path, logits_2d.astype(np.float32))
		print(f'[save] {dom}_{vw} -> {out_path} shape={logits_2d.shape}')

# -------------------------
# Final prints (per-view)
# -------------------------

print('\n=== Per-view report (Hit@ms) ===')
for dom, vw in view_name_list:
	key = f'{dom}_{vw}'
	if key in per_view_report:
		r = per_view_report[key]
		print(
			f'{key:>11}: '
			+ ' '.join([f'hit{int(t)}={r.get(f"hit@{int(t)}", 0):.3f}' for t in thr_ms])
			+ f' (n={r.get("n_tr_valid", 0)})'
		)

# -------------------------
# Cross-domain merge (raw-global row-wise)
# -------------------------

# use all available (domain,view) that were computed
cross_use = []
for dom in domains:
	if 'none' in views_cfg:
		cross_use.append((dom, 'none'))
	if 'hflip' in views_cfg:
		cross_use.append((dom, 'hflip'))

if fb_full is not None and dt_full is not None:
	if cross_method == 'logits_sum':
		cross_res = eval_cross_logits_sum(
			all_logits,
			fb_full,
			dt_full,
			thr_ms,
			use=cross_use,
			view_weights=cross_weights,
		)
		print(
			'\n[cross-merge logits_sum] '
			+ ' '.join(
				[f'hit{int(t)}={cross_res.get(f"hit@{int(t)}", 0):.3f}' for t in thr_ms]
			)
			+ f' (n={cross_res.get("n_tr_valid", 0)})'
		)
	else:  # default PoE
		cross_res = eval_cross_poe(
			all_logits,
			fb_full,
			dt_full,
			thr_ms,
			use=cross_use,
			view_weights=cross_weights,
			view_temps=cross_temps,
		)
		print(
			'\n[cross-merge PoE] '
			+ ' '.join(
				[f'hit{int(t)}={cross_res.get(f"hit@{int(t)}", 0):.3f}' for t in thr_ms]
			)
			+ f' (n={cross_res.get("n_tr_valid", 0)})'
		)
else:
	print('\n[cross-merge] skipped (no logits/fb collected)')

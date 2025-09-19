# %%
# proc/test.inference_tta.py
"""Collect ALL logits per view and evaluate per view (6 views total).

- Views: horizontal flip only -> {none, hflip}
- Domains: shot(=ffid), recv(=chno), cmp(=cmp)
- Loop order: for domain in domains: for view in [none, hflip]
- For each (domain, view):
    * run model without internal TTA
    * invert flip so logits return to the **original trace order**
    * reconstruct a single (num_traces_total, W) array by averaging overlapped windows
    * evaluate Hit@{0,2,4,8} ms on this view
    * (optionally) save that 2D array to disk: <out_root>/<domain>_<view>.npy
- After all 6 views are processed:
    * build all_logits = [shot_none, shot_hflip, chno_none, chno_hflip, cmp_none, cmp_hflip]

No cross-domain/merged evaluation here (per your request).
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
from proc.util.features import make_offset_channel_phys, make_time_channel
from proc.util.model import NetAE, adjust_first_conv_padding
from proc.util.model_utils import inflate_input_convs_to_2ch
from proc.util.utils import collect_field_files, set_seed

# -------------------------
# Minimal model helpers
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
	if getattr(cfg.model, 'use_offset_input', False):
		inflate_input_convs_to_2ch(model)
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
	return x.flip(dims=(2,))  # traces axis


def apply_view(x: torch.Tensor, view: TTAName) -> torch.Tensor:
	return x if view == 'none' else _hflip(x)


def invert_view(x: torch.Tensor, view: TTAName) -> torch.Tensor:
	return x if view == 'none' else _hflip(x)


# -------------------------
# Deterministic inference enumerator (windows)
# -------------------------


class InferenceGatherWindows(MaskedSegyGather):
	"""Deterministic window enumerator for inference (no random aug)."""

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

		x = info['mmap'][idx_win].astype(np.float32)
		x = x - x.mean(axis=1, keepdims=True)
		x = x / (x.std(axis=1, keepdims=True) + 1e-10)

		x, _ = self._fit_time_len(x, start=0)
		W = x.shape[1]

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
		]:
			meta[k] = torch.stack([b[k] for b in batch])
		meta['group_id'] = [b['group_id'] for b in batch]
		meta['domain'] = b0['domain']
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
# Per-view accumulator (flatten to (num_traces_total, W))
# -------------------------


class ViewAccumulator:
	"""Accumulate logits for one (domain, view) and evaluate once after the whole pass."""

	def __init__(self):
		self.sum_by_gid: dict[str, np.ndarray] = {}
		self.cnt_by_gid: dict[str, np.ndarray] = {}
		self.fb_by_gid: dict[str, np.ndarray] = {}
		self.dt_by_gid: dict[str, np.ndarray] = {}
		self.order: list[str] = []  # appearance order
		self.W: int | None = None

	def _ensure_gid(self, gid: str, Htot: int, W: int, dt_b: float):
		if gid not in self.sum_by_gid:
			self.sum_by_gid[gid] = np.zeros((Htot, W), dtype=np.float64)
			self.cnt_by_gid[gid] = np.zeros((Htot,), dtype=np.int32)
			self.fb_by_gid[gid] = np.full((Htot,), -1, dtype=np.int64)
			self.dt_by_gid[gid] = np.full((Htot,), dt_b, dtype=np.float32)
			self.order.append(gid)
		else:
			# Htot/W の整合性チェック
			H_prev, W_prev = self.sum_by_gid[gid].shape
			if H_prev != Htot or W_prev != W:
				raise RuntimeError(
					f'Inconsistent H/W for gid={gid}: ({H_prev},{W_prev}) vs ({Htot},{W})'
				)
			# dt は per-trace 上書き可（同じ値のはず）
		if self.W is None:
			self.W = int(W)
		elif int(W) != self.W:
			raise RuntimeError(f'Inconsistent W across groups: {self.W} vs {W}')

	def add_batch(self, meta: dict, logits: torch.Tensor, *, view_name: str):
		"""logits: (B,1,Hwin,W) already inverted to original order for this view."""
		arr = logits.to(torch.float32).detach().cpu().numpy()
		B, _, Hwin, W = arr.shape

		fb_idx = meta['fb_idx'].numpy()  # (B,Hwin)
		dt_sec = meta['dt_sec'].numpy().reshape(-1)  # (B,)
		abs_h = meta['abs_h'].numpy()  # (B,Hwin)
		gids = meta['group_id']
		Htot_b = meta['gather_len'].numpy().reshape(-1)  # (B,)

		for b in range(B):
			gid = str(gids[b])
			Htot = int(Htot_b[b])
			dt_b = float(dt_sec[b])
			self._ensure_gid(gid, Htot, W, dt_b)

			sum_mat = self.sum_by_gid[gid]
			cnt_vec = self.cnt_by_gid[gid]
			fb_vec = self.fb_by_gid[gid]
			dt_vec = self.dt_by_gid[gid]

			for h in range(Hwin):
				pos = int(abs_h[b, h])
				if pos < 0:  # padded
					continue
				sum_mat[pos, :] += arr[b, 0, h, :]
				cnt_vec[pos] += 1
				# fb/dt の更新
				fb_vec[pos] = int(fb_idx[b, h])
				dt_vec[pos] = dt_b

	def finalize_logits(self) -> np.ndarray:
		"""Return (N_total_traces, W) averaged logits in the first-seen gid order."""
		mats = []
		for gid in self.order:
			sum_mat = self.sum_by_gid[gid]
			cnt_vec = self.cnt_by_gid[gid]
			avg = sum_mat.copy()
			nz = cnt_vec > 0
			if nz.any():
				avg[nz, :] = (avg[nz, :].T / cnt_vec[nz]).T
			mats.append(avg.astype(np.float32))
		if not mats:
			return np.zeros((0, int(self.W or 0)), dtype=np.float32)
		return np.concatenate(mats, axis=0)

	def finalize_labels(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return (fb_idx_flat, dt_sec_flat) aligned to finalize_logits() order."""
		fbs, dts = [], []
		for gid in self.order:
			fbs.append(self.fb_by_gid[gid].astype(np.int64))
			dts.append(self.dt_by_gid[gid].astype(np.float32))
		if not fbs:
			return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
		return np.concatenate(fbs), np.concatenate(dts)


def eval_view(
	logits_2d: np.ndarray,
	fb_idx: np.ndarray,
	dt_sec: np.ndarray,
	thr_ms: Sequence[float],
) -> dict:
	"""Evaluate Hit@{thr_ms} for one view. logits_2d: (N, W)."""
	if logits_2d.size == 0:
		return {f'hit@{int(t)}': 0.0 for t in thr_ms} | {'n_tr_valid': 0}
	W = logits_2d.shape[1]
	pred_idx = logits_2d.argmax(axis=1).astype(
		np.int64
	)  # argmax(logit) == argmax(sigmoid(logit))
	valid = (fb_idx > 0) & (fb_idx < W)
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
model.use_tta = False  # disable model internal TTA

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

# -------------------------
# Main loop: per view
# -------------------------


def run_one_view(domain: str, view: TTAName) -> tuple[np.ndarray, dict]:
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

	model.eval()
        for x, _t, _m, meta in loader:
                x = x.to(device, non_blocking=True)
                # build input (offset channel if requested)
                x_in = x
                use_offset = getattr(cfg.model, 'use_offset_input', False)
                use_time = getattr(cfg.model, 'use_time_input', False)
                if (use_offset or use_time) and ('offsets' in meta):
                        offs = make_offset_channel_phys(
                                x, meta['offsets'],
                                x95_m=cfg.norm.x95_m,
                                mode=getattr(cfg.norm, 'offset_mode', 'log1p'),
                                clip_hi=getattr(cfg.norm, 'offset_clip_hi', 1.5),
                        ).to(device=x.device, dtype=x.dtype)
                        if use_time and ('dt_sec' in meta):
                                time_ch = make_time_channel(
                                        x, meta['dt_sec'],
                                        t95_ms=cfg.norm.t95_ms,
                                        clip_hi=getattr(cfg.norm, 'time_clip_hi', 1.5),
                                ).to(device=x.device, dtype=x.dtype)
                                x_in = torch.cat([x, offs, time_ch], dim=1)
                        else:
                                x_in = torch.cat([x, offs], dim=1)

		# run single view
		dev_type = 'cuda' if x.is_cuda else 'cpu'
		with autocast(device_type=dev_type, enabled=use_amp):
			xv = apply_view(x_in, view)
			logit = model(xv)  # (B,1,H,W)
			logit = invert_view(logit, view)  # back to original trace order

		acc.add_batch(meta, logit, view_name=view)
		pbar.update(1)

	pbar.close()

	# finalize & evaluate
	logits_2d = acc.finalize_logits()  # (N, W)
	fb_idx, dt_sec = acc.finalize_labels()
	report = eval_view(logits_2d, fb_idx, dt_sec, thr_ms)
	return logits_2d, report


for dom, vw in view_name_list:
	if dom not in domains:
		continue
	logits_2d, rep = run_one_view(dom, vw)
	idx = view_name_list.index((dom, vw))
	all_logits[idx] = logits_2d
	per_view_report[f'{dom}_{vw}'] = rep

	# save only the 2D logits per view (no UID分割保存)
	if save_logits:
		out_path = Path(out_root_cfg) / f'{dom}_{vw}.npy'
		np.save(out_path, logits_2d.astype(np.float32))
		print(f'[save] {dom}_{vw} -> {out_path} shape={logits_2d.shape}')

# -------------------------
# Final prints
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

# all_logits is now:
# [0]: shot_none, [1]: shot_hflip, [2]: chno_none, [3]: chno_hflip, [4]: cmp_none, [5]: cmp_hflip

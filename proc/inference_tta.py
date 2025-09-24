# %%
# proc/test.inference_tta.py
"""Test-Time Augmentation (TTA) inference & validation for FB segmentation with Hydra config.

- Augmentations: **horizontal flip only** (trace order). *Vertical (time) flip is intentionally disabled.*
- Domains: shot-gather (ffid), receiver-gather (chno), CMP-gather (cmp), and super-gather (superwindow over shot)
- Per-domain metrics + optional **cross-domain merge** over the entire survey
- TTA merge rule (within a view set): mean | median (default: mean)
- Cross-domain merge rule: mean | median | wmean (confidence-weighted mean)
- Metric: Hit@{0,2,4,8} ms (fraction of traces whose predicted FB falls within tolerance)

Usage
-----
python -m proc.test.inference_tta \
  infer.ckpt=/path/to/ckpt.pt \
  infer.tta_views="[none,hflip]" infer.merge=mean \
  infer.cross_merge=true infer.cross_merge_mode=median infer.cross_min_domains=2 \
  infer.id_key_pairs='[[ffid,chno],[chno,ffid],[cmp,offset]]'

Notes
-----
* cross-merge対象ドメインは **未指定なら `domains` から自動導出**（既定で `super` を除外）。
  必要な場合のみ `infer.cross_merge_domains` で明示的に上書きしてください。
* キーの組（UID生成）は `infer.id_key_pairs` で優先順を指定（既定: ffid-chno → chno-ffid → cmp-offset）。

"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

import torch
from hydra import main
from omegaconf import DictConfig, OmegaConf
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, SequentialSampler

from proc.util.dataset import MaskedSegyGather
from proc.util.features import make_offset_channel_phys, make_time_channel
from proc.util.model import NetAE, adjust_first_conv_padding
from proc.util.model_utils import inflate_input_convs_to_nch, inflate_input_convs_to_2ch
from proc.util.utils import collect_field_files, set_seed
from proc.util.velocity_mask import (
	apply_velocity_mask_to_logits,
	make_velocity_feasible_mask,
)

# -------------------------
# Utilities
# -------------------------


def build_model_from_cfg(cfg) -> torch.nn.Module:
	"""Instantiate model for FB segmentation / recon heads."""
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
			model,
			3,
			verbose=True,
			init_mode='duplicate',
		)
	elif getattr(cfg.model, 'use_offset_input', False):
		inflate_input_convs_to_2ch(
			model,
			verbose=True,
			init_mode='duplicate',
		)

	return model


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
	state = torch.load(str(path), map_location='cpu', weights_only=False)
	cand_keys = ['model_ema', 'state_dict', 'model']
	for k in cand_keys:
		if k in state and isinstance(state[k], dict):
			missing, unexpected = model.load_state_dict(state[k], strict=False)
			print(
				f"[ckpt] loaded via key='{k}': missing={len(missing)} unexpected={len(unexpected)}"
			)
			return
	missing, unexpected = model.load_state_dict(state, strict=False)
	print(
		f'[ckpt] loaded raw dict: missing={len(missing)} unexpected={len(unexpected)}'
	)


# Flip helpers ---------------------------------------------------------------

TTAName = Literal['none', 'hflip']


def _hflip(x: torch.Tensor) -> torch.Tensor:
	return x.flip(dims=(2,))  # traces axis


FLIP_FNS: dict[
	TTAName,
	tuple[
		Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
	],
] = {
	'none': (lambda x: x, lambda x: x),
	'hflip': (_hflip, _hflip),
}


def _merge_stack(
	x_list: list[torch.Tensor], mode: Literal['mean', 'median']
) -> torch.Tensor:
	x = torch.stack(x_list, dim=0)
	if mode == 'mean':
		return x.mean(dim=0)
	if mode == 'median':
		return x.median(dim=0).values
	raise ValueError(f'Invalid merge mode: {mode}')


# Prediction core ------------------------------------------------------------


@torch.no_grad()
def predict_with_tta(
        model: torch.nn.Module,
        x: torch.Tensor,
        meta: dict,
        cfg,
        *,
        device: torch.device,
        tta_views: Iterable[TTAName] = ('none', 'hflip'),
        merge: Literal['mean', 'median'] = 'mean',
        use_amp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pred_idx, prob_max) for a batch."""
        B, C, H, W = x.shape
        x_in = x
        use_offset = getattr(cfg.model, 'use_offset_input', False)
        use_time = getattr(cfg.model, 'use_time_input', False)
        if (use_offset or use_time) and ('offsets' in meta):
                offs_ch = make_offset_channel_phys(
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
                        x_in = torch.cat([x, offs_ch, time_ch], dim=1)
                else:
                        x_in = torch.cat([x, offs_ch], dim=1)

	tta_logits = []
	dev_type = 'cuda' if x.is_cuda else 'cpu'
	with autocast(device_type=dev_type, enabled=use_amp):
		for name in tta_views:
			f, inv = FLIP_FNS[name]
			xin = f(x_in)
			logits = model(xin)  # (B,1,H,W)
			logits = inv(logits)
			tta_logits.append(logits)

	logit = _merge_stack(tta_logits, mode=merge)  # (B,1,H,W)

	fb_cfg = getattr(cfg.loss, 'fb_seg', None)
	if (
		('offsets' in meta)
		and ('dt_sec' in meta)
		and bool(getattr(cfg.infer, 'use_velocity_mask', True))
		and (fb_cfg is not None)
	):
		velmask = make_velocity_feasible_mask(
			offsets=meta['offsets'].to(device),
			dt_sec=meta['dt_sec'].to(device).view(B),
			W=W,
			vmin=float(getattr(fb_cfg, 'vmin_mask', 500.0)),
			vmax=float(getattr(fb_cfg, 'vmax_mask', 10000.0)),
			t0_lo_ms=float(getattr(fb_cfg, 't0_lo_ms', -100.0)),
			t0_hi_ms=float(getattr(fb_cfg, 't0_hi_ms', 80.0)),
			taper_ms=float(getattr(fb_cfg, 'taper_ms', 10.0)),
			device=device,
		)
		logit = apply_velocity_mask_to_logits(logit, velmask)

	prob = torch.sigmoid(logit[:, 0])  # (B,H,W)
	prob_max, pred_idx = prob.max(dim=-1)  # (B,H)
	return pred_idx, prob_max


# Per-domain evaluation ------------------------------------------------------


@torch.no_grad()
def eval_fbseg_with_tta(
	model: torch.nn.Module,
	loader: DataLoader,
	cfg,
	*,
	device: torch.device,
	tta_views: Iterable[TTAName] = ('none', 'hflip'),
	merge: Literal['mean', 'median'] = 'mean',
	use_amp: bool = True,
) -> dict:
	"""Per-domain evaluation (Hit@ms)."""
	model.eval()
	fb_cfg = getattr(cfg.loss, 'fb_seg', None)
	if fb_cfg is None:
		raise RuntimeError(
			'cfg.loss.fb_seg not found; set task=fb_seg / provide loss.fb_seg in config'
		)

	thr_ms = getattr(fb_cfg, 'hit_thresholds_ms', (0.0, 2.0, 4.0, 8.0))
	thr_ms = tuple(float(t) for t in thr_ms)

	hit_counts = {f'hit@{int(t)}': 0 for t in thr_ms}
	n_valid_total = 0

	for x, _teacher, _mask, meta in loader:
		x = x.to(device, non_blocking=True)
		B, C, H, W = x.shape

		pred_idx, _prob_max = predict_with_tta(
			model,
			x,
			meta,
			cfg,
			device=device,
			tta_views=tta_views,
			merge=merge,
			use_amp=use_amp,
		)

		fb_idx = meta['fb_idx'].to(device)
		valid = (fb_idx > 0) & (fb_idx < W)
		dt_sec = meta['dt_sec'].to(device).view(B)
		tol_samp = [
			torch.round(torch.tensor(t / 1000.0, device=device) / dt_sec).to(torch.long)
			for t in thr_ms
		]
		fb_idx = fb_idx.clamp(0, W - 1)

		diff = (pred_idx - fb_idx).abs()
		n_valid_total += int(valid.sum().item())

		for t, tol in zip(thr_ms, tol_samp, strict=False):
			tol_bh = tol.view(B, 1).expand_as(diff)
			hit = (diff <= tol_bh) & valid
			hit_counts[f'hit@{int(t)}'] += int(hit.sum().item())

	out = {k: (v / max(n_valid_total, 1)) for k, v in hit_counts.items()}
	out['n_tr_valid'] = n_valid_total
	return out


# Cross-domain aggregator ----------------------------------------------------


class KeyResolver:
	"""Build a canonical UID per trace from configured key-pairs.

	Default order tries: (ffid,chno) -> (chno,ffid) -> (cmp,offset).
	This is easily extensible via infer.id_key_pairs.
	"""

	def __init__(self, key_pairs: Sequence[Sequence[str]]):
		# normalize to tuples of two strings
		self.pairs: list[tuple[str, str]] = [(str(a), str(b)) for a, b in key_pairs]

	def get_uid(self, meta: dict, b: int, h: int) -> str | None:
		def _get(name: str):
			if name == 'offset':
				# derive scalar per trace from 'offsets'
				if 'offsets' in meta:
					return float(meta['offsets'][b, h].item())
				return None
			if name in meta:
				return int(meta[name][b, h].item())
			return None

		for k1, k2 in self.pairs:
			v1, v2 = _get(k1), _get(k2)
			if (v1 is not None) and (v2 is not None):
				return f'{k1}={v1}|{k2}={v2}'
		return None


class CrossDomainAggregator:
	"""Accumulate per-trace predictions across domains and merge globally.

	Keying uses the KeyResolver; within-domain duplicates are reduced by median.
	"""

	def __init__(
		self,
		key_pairs: Sequence[Sequence[str]],
		cross_mode: Literal['mean', 'median', 'wmean'] = 'median',
		min_domains: int = 2,
	):
		self.resolver = KeyResolver(key_pairs)
		self.cross_mode = cross_mode
		self.min_domains = int(min_domains)
		self.pred: dict[str, dict[str, list[int]]] = defaultdict(
			lambda: defaultdict(list)
		)  # uid -> domain -> [idx]
		self.conf: dict[str, dict[str, list[float]]] = defaultdict(
			lambda: defaultdict(list)
		)  # uid -> domain -> [pmax]
		self.gt: dict[str, int] = {}
		self.dt: dict[str, float] = {}

	def add_batch(
		self,
		domain: str,
		pred_idx: torch.Tensor,
		prob_max: torch.Tensor,
		meta: dict,
		W: int,
	):
		B, H = pred_idx.shape
		fb_idx = meta.get('fb_idx')
		dt_sec = meta.get('dt_sec')
		for b in range(B):
			dt_b = float(dt_sec[b].item()) if dt_sec is not None else float('nan')
			for h in range(H):
				uid = self.resolver.get_uid(meta, b, h)
				if uid is None:
					continue  # cannot key this trace
				pidx = int(pred_idx[b, h].item())
				pmax = float(prob_max[b, h].item())
				self.pred[uid][domain].append(pidx)
				self.conf[uid][domain].append(pmax)
				if fb_idx is not None:
					self.gt[uid] = int(min(max(int(fb_idx[b, h].item()), 0), W - 1))
				if not math.isnan(dt_b):
					self.dt.setdefault(uid, dt_b)

	def finalize(self, thr_ms: Iterable[float]) -> dict:
		hits = {f'hit@{int(t)}': 0 for t in thr_ms}
		n_valid = 0
		for uid, by_dom in self.pred.items():
			# reduce within-domain first
			dom_vals = {}
			dom_weights = {}
			for d, vals in by_dom.items():
				if not vals:
					continue
				if self.cross_mode == 'mean':
					dom_vals[d] = sum(vals) / len(vals)
				elif self.cross_mode == 'wmean':
					ws = self.conf[uid][d]
					wsum = sum(ws) + 1e-8
					dom_vals[d] = (
						sum(v * w for v, w in zip(vals, ws, strict=False)) / wsum
					)
				else:  # median
					s = sorted(vals)
					m = (
						s[len(s) // 2]
						if len(s) % 2 == 1
						else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
					)
					dom_vals[d] = m
				dom_weights[d] = len(vals)

			if len(dom_vals) < self.min_domains:
				continue

			# cross-domain merge
			vals = list(dom_vals.values())
			if self.cross_mode == 'mean':
				merged = round(sum(vals) / len(vals))
			elif self.cross_mode == 'wmean':
				weights = [dom_weights[d] for d in dom_vals]
				merged = round(
					sum(v * w for v, w in zip(vals, weights, strict=False))
					/ (sum(weights) + 1e-8)
				)
			else:
				s = sorted(vals)
				merged = round(
					s[len(s) // 2]
					if len(s) % 2 == 1
					else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
				)

			# evaluate
			if uid not in self.gt or uid not in self.dt:
				continue
			gt = self.gt[uid]
			dt = self.dt[uid]
			diff = abs(int(merged) - int(gt))
			n_valid += 1
			for t in thr_ms:
				tol = int(round((t / 1000.0) / dt))
				if diff <= tol:
					hits[f'hit@{int(t)}'] += 1
		out = {k: (v / max(n_valid, 1)) for k, v in hits.items()}
		out['n_tr_valid'] = n_valid
		return out


# Domain loaders -------------------------------------------------------------


def make_valid_loader_for_domain(
	cfg,
	*,
	domain: Literal['shot', 'recv', 'cmp', 'super'],
	batch_size: int | None = None,
	num_workers: int | None = None,
) -> DataLoader:
	data_root = Path(cfg.data_root)
	vlist_name = getattr(cfg, 'valid_field_list', None) or getattr(
		cfg.dataset, 'valid_field_list', None
	)
	if vlist_name is None:
		raise RuntimeError(
			'valid_field_list is not set. Define it in inference.yaml or base.yaml'
		)
	valid_field_list = Path(__file__).parent / 'configs' / str(vlist_name)

	segy_files, fb_files = collect_field_files(
		data_root, valid_field_list, file_size=cfg.file_size
	)

	use_super = domain == 'super'
	if domain == 'shot':
		pk = ('ffid',)
	elif domain == 'recv':
		pk = ('chno',)
	elif domain == 'cmp':
		pk = ('cmp',)
	elif domain == 'super':
		pk = ('ffid',)
	else:
		raise ValueError(f'unknown domain: {domain}')

	ds = MaskedSegyGather(
		segy_files,
		fb_files,
		use_header_cache=getattr(cfg.dataset, 'use_header_cache', False),
		header_cache_dir=getattr(cfg.dataset, 'header_cache_dir', None),
		primary_keys=pk,
		primary_key_weights=None,
		use_superwindow=bool(
			getattr(cfg.dataset, 'use_superwindow', False) or use_super
		),
		sw_halfspan=int(getattr(cfg.dataset, 'sw_halfspan', 0)),
		sw_prob=1.0,
		mask_ratio=float(getattr(cfg.dataset, 'mask_ratio', 0.0)),
		mask_mode=str(getattr(cfg.dataset, 'mask_mode', 'replace')),
		mask_noise_std=float(getattr(cfg.dataset, 'mask_noise_std', 1.0)),
		pick_ratio=float(getattr(cfg.dataset, 'pick_ratio', 0.3)),
		target_len=int(getattr(cfg.dataset, 'target_len', 6016)),
		flip=False,
		augment_time_prob=0.0,
		augment_space_prob=0.0,
		augment_freq_prob=0.0,
		target_mode='fb_seg',
		label_sigma=float(getattr(cfg.dataset, 'label_sigma', 1.0)),
		reject_fblc=False,
	)

	bs = int(batch_size or getattr(cfg.infer, 'batch_size', cfg.batch_size))
	nw = int(
		num_workers or getattr(cfg.infer, 'num_workers', getattr(cfg, 'num_workers', 4))
	)

	def _collate(batch):
		if isinstance(batch[0], dict) and 'x' in batch[0]:
			x = torch.stack([b['x'] for b in batch])
			meta = {}
			for k in ['fb_idx', 'offsets', 'dt_sec', 'ffid', 'chno', 'cmp']:
				if k in batch[0]:
					meta[k] = torch.stack([b[k] for b in batch])
			return x, None, None, meta
		return batch[0]

	loader = DataLoader(
		ds,
		batch_size=bs,
		sampler=SequentialSampler(ds),
		num_workers=nw,
		pin_memory=True,
		drop_last=False,
		collate_fn=_collate,
	)
	return loader


# Hydra main ----------------------------------------------------------------


@main(config_path='configs', config_name='inference', version_base='1.3')
def _main(cfg: DictConfig) -> None:
	print('[cfg]' + OmegaConf.to_yaml(cfg))
	set_seed(int(getattr(cfg, 'seed', 42)))

	device = torch.device(
		str(getattr(cfg, 'device', 'cuda')) if torch.cuda.is_available() else 'cpu'
	)
	use_amp = bool(getattr(cfg, 'use_amp', True))

	model = build_model_from_cfg(cfg).to(device)
	ckpt = getattr(cfg.infer, 'ckpt', None)
	if ckpt is None:
		raise RuntimeError('Please set infer.ckpt to a checkpoint path')
	load_checkpoint(model, ckpt)
	model.eval()

	# TTA view set from config (only none/hflip allowed)
	tta_views = tuple(getattr(cfg.infer, 'tta_views', ['none', 'hflip']))
	allowed = {'none', 'hflip'}
	bad = [v for v in tta_views if v not in allowed]
	if bad:
		raise RuntimeError(
			f'infer.tta_views contains unsupported entries (no vflip allowed): {bad}'
		)

	merge = str(getattr(cfg.infer, 'merge', 'mean'))
	domains = list(getattr(cfg.infer, 'domains', ['shot', 'recv', 'cmp', 'super']))

	# cross-domain settings
	cross_enable = bool(getattr(cfg.infer, 'cross_merge', True))
	cross_mode = str(
		getattr(cfg.infer, 'cross_merge_mode', 'median')
	)  # mean|median|wmean
	cross_min_domains = int(getattr(cfg.infer, 'cross_min_domains', 2))
	# auto-derive cross_merge_domains from `domains` (exclude 'super' by default)
	_cm_domains = getattr(cfg.infer, 'cross_merge_domains', None)
	cross_domains = (
		[d for d in domains if d != 'super']
		if (_cm_domains is None or _cm_domains == [])
		else list(_cm_domains)
	)
	id_key_pairs = getattr(
		cfg.infer,
		'id_key_pairs',
		[['ffid', 'chno'], ['chno', 'ffid'], ['cmp', 'offset']],
	)

	# thresholds for reporting
	fb_cfg = getattr(cfg.loss, 'fb_seg', None)
	thr_ms = tuple(
		float(t) for t in getattr(fb_cfg, 'hit_thresholds_ms', (0.0, 2.0, 4.0, 8.0))
	)

	summary = {}
	total_valid = 0
	micro_hits = defaultdict(int)

	aggregator = (
		CrossDomainAggregator(id_key_pairs, cross_mode, min_domains=cross_min_domains)
		if cross_enable
		else None
	)

	for domain in domains:
		print(f'[domain={domain}] building loader…')
		loader = make_valid_loader_for_domain(cfg, domain=domain)
		max_batches = int(getattr(cfg.infer, 'max_batches', 0))
		if max_batches > 0:
			it = iter(loader)

			def limited_iter():
				for _ in range(max_batches):
					try:
						yield next(it)
					except StopIteration:
						return

			eval_loader = limited_iter()
		else:
			eval_loader = loader

		print(f'[domain={domain}] running TTA={tta_views} merge={merge}')

		# per-domain eval while optionally accumulating for cross-merge
		model.eval()
		hit_counts = {f'hit@{int(t)}': 0 for t in thr_ms}
		n_valid_total = 0

		for x, _teacher, _mask, meta in eval_loader:
			x = x.to(device, non_blocking=True)
			B, C, H, W = x.shape

			pred_idx, prob_max = predict_with_tta(
				model,
				x,
				meta,
				cfg,
				device=device,
				tta_views=tta_views,
				merge=merge,
				use_amp=use_amp,
			)

			fb_idx = meta['fb_idx'].to(device)
			valid = (fb_idx > 0) & (fb_idx < W)
			dt_sec = meta['dt_sec'].to(device).view(B)
			tol_samp = [
				torch.round(torch.tensor(t / 1000.0, device=device) / dt_sec).to(
					torch.long
				)
				for t in thr_ms
			]
			fb_idx = fb_idx.clamp(0, W - 1)

			diff = (pred_idx - fb_idx).abs()
			n_valid_total += int(valid.sum().item())

			for t, tol in zip(thr_ms, tol_samp, strict=False):
				tol_bh = tol.view(B, 1).expand_as(diff)
				hit = (diff <= tol_bh) & valid
				hit_counts[f'hit@{int(t)}'] += int(hit.sum().item())

			# accumulate for cross-domain merge if this domain is selected
			if aggregator is not None and (domain in cross_domains):
				aggregator.add_batch(domain, pred_idx.cpu(), prob_max.cpu(), meta, W)

		res = {k: (v / max(n_valid_total, 1)) for k, v in hit_counts.items()}
		res['n_tr_valid'] = n_valid_total

		summary[domain] = res
		total_valid += int(res.get('n_tr_valid', 0))
		for k in [k for k in res if k.startswith('hit@')]:
			micro_hits[k] += int(round(res[k] * max(res.get('n_tr_valid', 0), 1)))
		print(f'[domain={domain}] -> {res}')

	micro_avg = {k: (v / max(total_valid, 1)) for k, v in micro_hits.items()}

	print('=== Summary ===')
	for d, r in summary.items():
		print(
			f'{d:>7}: hit0={r.get("hit@0", 0):.3f} hit2={r.get("hit@2", 0):.3f} hit4={r.get("hit@4", 0):.3f} hit8={r.get("hit@8", 0):.3f} (n={r.get("n_tr_valid", 0)})'
		)
	print(
		f'micro : hit0={micro_avg.get("hit@0", 0):.3f} hit2={micro_avg.get("hit@2", 0):.3f} hit4={micro_avg.get("hit@4", 0):.3f} hit8={micro_avg.get("hit@8", 0):.3f} (n={total_valid})'
	)

	# cross-domain merge result over entire survey
	if aggregator is not None:
		cross_res = aggregator.finalize(thr_ms)
		print(
			'[cross-merge] domains='
			+ ','.join(cross_domains)
			+ f' mode={cross_mode} min_domains={cross_min_domains}'
		)
		print(
			f'cross : hit0={cross_res.get("hit@0", 0):.3f} hit2={cross_res.get("hit@2", 0):.3f} hit4={cross_res.get("hit@4", 0):.3f} hit8={cross_res.get("hit@8", 0):.3f} (n={cross_res.get("n_tr_valid", 0)})'
		)


if __name__ == '__main__':
	_main()  # hydra entry

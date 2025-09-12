# %%
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure we can import project modules when this file sits next to train.py
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
	sys.path.insert(0, str(_THIS_DIR))

# Project-relative imports (same as train.py)
from util.dataset import MaskedSegyGather  # type: ignore
from util.utils import collect_field_files, set_seed  # type: ignore

set_seed(0)


def _ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def _fig_save(fig, out_png: Path) -> None:
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_png, dpi=150, bbox_inches='tight')
	plt.close(fig)


def _fb_to_map(fb_idx: np.ndarray, H: int, W: int):
	fbmap = np.zeros((H, W), dtype=np.float32)
	valid = (fb_idx >= 0) & (fb_idx < W)
	rows = np.where(valid)[0]
	cols = fb_idx[valid]
	fbmap[rows, cols] = 1.0
	return fbmap, valid


def _neighbor_stats(fb_idx: np.ndarray, dt_sec: float) -> dict[str, float]:
	valid = fb_idx >= 0
	v = fb_idx.astype(np.float64)
	m = valid[:-1] & valid[1:]
	if not np.any(m):
		return dict(
			count_pairs=0,
			var_samples=np.nan,
			p95_samples=np.nan,
			var_ms=np.nan,
			p95_ms=np.nan,
		)
	diffs = np.abs(v[1:] - v[:-1])[m]
	var_samp = float(np.var(diffs)) if diffs.size > 1 else 0.0
	p95_samp = float(np.percentile(diffs, 95)) if diffs.size > 0 else 0.0
	diffs_ms = diffs * (dt_sec * 1000.0)
	var_ms = float(np.var(diffs_ms)) if diffs_ms.size > 1 else 0.0
	p95_ms = float(np.percentile(diffs_ms, 95)) if diffs_ms.size > 0 else 0.0
	return dict(
		count_pairs=int(m.sum()),
		var_samples=var_samp,
		p95_samples=p95_samp,
		var_ms=var_ms,
		p95_ms=p95_ms,
	)


def _build_ds_kwargs_from_cfg(cfg: Any) -> dict[str, Any]:
	"""Pick dataset-related fields from a Hydra-like cfg (OmegaConf or dict)."""
	dcfg = getattr(cfg, 'dataset', cfg)  # allow cfg.dataset.* or flat dict

	def g(name, default=None):
		if hasattr(dcfg, name):
			return getattr(dcfg, name)
		if isinstance(dcfg, dict):
			return dcfg.get(name, default)
		return default

	return dict(
		use_header_cache=bool(g('use_header_cache', False)),
		header_cache_dir=g('header_cache_dir', None),
		mask_ratio=float(g('mask_ratio', 0)),
		mask_mode=g('mask_mode', 'replace'),
		mask_noise_std=float(g('mask_noise_std', 1.0)),
		pick_ratio=float(g('pick_ratio', 0.3)),
		target_len=int(g('target_len', 6016)),
		flip=bool(g('flip', False)),
		augment_time_prob=float(g('augment_time_prob', 0.0)),
		augment_time_range=tuple(g('augment_time_range', (0.95, 1.05))),
		augment_space_prob=float(g('augment_space_prob', 0.0)),
		augment_space_range=tuple(g('augment_space_range', (0.90, 1.10))),
		augment_freq_prob=float(g('augment_freq_prob', 0.0)),
		augment_freq_kinds=tuple(
			g('augment_freq_kinds', ('bandpass', 'lowpass', 'highpass'))
		),
		augment_freq_band=tuple(g('augment_freq_band', (0.05, 0.45))),
		augment_freq_width=tuple(g('augment_freq_width', (0.10, 0.35))),
		augment_freq_roll=float(g('augment_freq_roll', 0.02)),
		augment_freq_restandardize=bool(g('augment_freq_restandardize', True)),
		target_mode=g('target_mode', 'fb_seg'),
		label_sigma=float(g('label_sigma', 1.0)),
		# ffid_byte / chno_byte / cmp_byte は dataset 側のデフォルトに任せる
		reject_fblc=bool(g('reject_fblc', False)),
		fblc_percentile=float(g('fblc_percentile', 95.0)),
		fblc_thresh_ms=float(g('fblc_thresh_ms', 8.0)),
		fblc_min_pairs=int(g('fblc_min_pairs', 16)),
		fblc_apply_on=g('fblc_apply_on', 'any'),
	)


def viz_gather(
	cfg: Any,
	field_list: str,
	data_root: str,
	domains: Sequence[str] = ('ffid', 'chno', 'cmp'),
	super_on: bool = False,
	sw_halfspan: int = 0,
	sw_prob: float = 0.3,
	n: int = 2,
	out: str = 'out_viz',
	seed: int = 1234,
) -> pd.DataFrame:
	"""Visualize & sanity-check samples across domains and settings.

	Parameters
	----------
	cfg : OmegaConf or dict
		Same config object you pass to train.py. Only dataset.* fields are read.
	field_list : str
		e.g., 'train_field_list.txt' under configs/ ; used by collect_field_files.
	data_root : str
		Root directory that contains subfolders named by each field in field_list.
	domains : list[str]
		Primary domains to test: subset of {'ffid','chno','cmp'}.
	super_on : bool
		Whether to enable superwindow.
	sw_halfspan : int
		Half-span (±) of primary keys to merge when super_on is True.
	sw_prob : float
		Probability gate to apply superwindow (0..1).
	n : int
		Number of samples per (domain, super) combo.
	out : str
		Output directory to save figures and CSV.
	seed : int
		RNG seed.

	Returns
	-------
	pandas.DataFrame : per-sample neighbor Δt statistics.

	"""
	np.random.seed(seed)
	outdir = Path(out)
	_ensure_dir(outdir)

	# 1) Same file selection as train.py
	segy_files, fb_files = collect_field_files(field_list, data_root)
	if len(segy_files) == 0:
		raise RuntimeError('No files collected. Check field_list/data_root.')
	print(f'Pairs: {len(segy_files)}  e.g., SEGY={segy_files[0]}  FB={fb_files[0]}')

	# 2) Dataset kwargs from cfg.dataset
	common_kwargs = _build_ds_kwargs_from_cfg(cfg)

	rows_out = []

	# 3) Iterate over domains
	for dom in domains:
		tag = f'{dom}_{"super" if super_on else "plain"}_h{sw_halfspan}_p{sw_prob}'
		print(f'\n=== {tag} ===')
		ds = MaskedSegyGather(
			segy_files=segy_files,
			fb_files=fb_files,
			primary_keys=(dom,),
			primary_key_weights=(1.0,),
			use_superwindow=bool(super_on),
			sw_halfspan=int(sw_halfspan),
			sw_prob=float(sw_prob),
			**common_kwargs,
		)

		for i in range(int(n)):
			s = ds[i]
			primary_unique = s.get('primary_unique', '')
			secondary_key = s.get('secondary_key', 'none')
			x = s['original'][0].detach().cpu().numpy()  # [H,W]
			fb_idx = s['fb_idx'].detach().cpu().numpy()  # [H]
			dt = float(s['dt_sec'])
			key_name = s.get('key_name', '?')
			H, W = x.shape

			fbmap, valid = _fb_to_map(fb_idx, H, W)
			stats = _neighbor_stats(fb_idx, dt)

			# --- Figure 1: gather with FB overlay ---
			fig1 = plt.figure(figsize=(8, 4.5))
			ax1 = fig1.add_subplot(111)
			ax1.imshow(x, aspect='auto', vmin=-3, vmax=3, origin='lower')
			rows_v = np.where(valid)[0]
			cols_v = fb_idx[valid]
			ax1.scatter(cols_v, rows_v, s=6, marker='o')
			ax1.set_title(
				f'gather [{key_name} | 2nd={secondary_key}] {tag} (H={H}, W={W}) | prim={primary_unique}'
			)
			ax1.set_xlabel('t [samples]')
			ax1.set_ylabel('trace index')
			_fig_save(fig1, outdir / f'gather_{tag}_{i:02d}.png')

			# --- Figure 2: first-break map (sparse 0/1) ---
			fig2 = plt.figure(figsize=(8, 4.0))
			ax2 = fig2.add_subplot(111)
			ax2.imshow(fbmap, aspect='auto', vmin=0, vmax=0.1, origin='lower')
			ax2.set_title('first-break map (1 at picked sample)')
			ax2.set_xlabel('t [samples]')
			ax2.set_ylabel('trace index')
			_fig_save(fig2, outdir / f'fbmap_{tag}_{i:02d}.png')

			# --- Figure 3: first-break curve ---
			fig3 = plt.figure(figsize=(8, 3.0))
			ax3 = fig3.add_subplot(111)
			tvals = np.arange(H)
			fbplot = np.where(valid, fb_idx, np.nan)
			ax3.plot(tvals, fbplot)
			ax3.set_title('first-break index per trace (NaN if invalid)')
			ax3.set_xlabel('trace index')
			ax3.set_ylabel('t [samples]')
			_fig_save(fig3, outdir / f'fbcurve_{tag}_{i:02d}.png')

			rows_out.append(
				dict(
					tag=tag,
					sample_idx=i,
					key_name=key_name,
					secondary_key=secondary_key,
					primary_unique=primary_unique,  # ← 追加
					count_pairs=stats['count_pairs'],
					var_samples=stats['var_samples'],
					p95_samples=stats['p95_samples'],
					var_ms=stats['var_ms'],
					p95_ms=stats['p95_ms'],
					file_path=s.get('file_path', '?'),
				)
			)

		ds.close()

	df = pd.DataFrame(rows_out)
	df.to_csv(outdir / 'neighbor_stats.csv', index=False)
	print(f'\nSaved figures and stats under: {outdir}')
	return df


from omegaconf import OmegaConf

cfg = OmegaConf.load('configs/base.yaml')

df = viz_gather(
	cfg=cfg,
	field_list=cfg.valid_field_list,
	data_root=cfg.data_root,
	domains=('ffid', 'chno', 'cmp'),
	super_on=True,
	sw_halfspan=1,
	sw_prob=1,
	n=10,
	out='out_viz',
	seed=1234,
)

df.head()

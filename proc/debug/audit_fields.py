"""Debug script to audit offset fields using audit utilities."""

# %%
# /workspace/proc/debug/audit_fields.py
"""Debug script to audit offset fields using audit utilities (no argparse)."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader, SequentialSampler

from proc.util.audit import audit_offsets_and_mask_coverage
from proc.util.collate import segy_collate
from proc.util.dataset import MaskedSegyGather
from proc.util.rng_util import worker_init_fn
from proc.util.velocity_mask import make_velocity_feasible_mask


def _list_path(list_name: str) -> Path:
	"""Find proc/configs/<list_name> first, then fallback to <repo_root>/configs."""
	here = Path(__file__).resolve()
	cand1 = here.parents[1] / 'configs' / list_name  # proc/configs/...
	cand2 = here.parents[2] / 'configs' / list_name  # <repo>/configs/...
	if cand1.exists():
		return cand1
	if cand2.exists():
		return cand2
	raise FileNotFoundError(
		f'field list not found: {list_name} (looked in {cand1} and {cand2})'
	)


def _collect_field_files(list_name: str, data_root: str):
	"""Return matching SEG-Y and FB files for each field listed."""
	lp = _list_path(list_name)
	with lp.open() as f:
		fields = [
			ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')
		]

	segy_files, fb_files = [], []
	for field in fields:
		d = Path(data_root) / field
		segys = sorted(list(d.glob('*.sgy')) + list(d.glob('*.segy')))
		fbs = sorted(d.glob('*.npy'))
		if not segys or not fbs:
			print(f'[WARN] skip {field}: missing SEG-Y or FB')
			continue
		segy_files.append(segys[0])
		fb_files.append(fbs[0])
	return segy_files, fb_files


def _audit_one_field(segy: Path, fb: Path, cfg, *, max_batches: int, cov_th: float):
	"""Run audit for a single (segy, fb) pair and return stats dict."""
	ds = MaskedSegyGather(
		[segy],
		[fb],
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
	loader = DataLoader(
		ds,
		batch_size=cfg.batch_size,
		sampler=SequentialSampler(ds),
		shuffle=False,
		num_workers=0,
		pin_memory=True,
		collate_fn=segy_collate,
		drop_last=False,
		worker_init_fn=worker_init_fn,
	)

	# 基本の監査（dx, coverage など）
	stats = audit_offsets_and_mask_coverage(
		loader, cfg.loss.fb_seg, max_batches=max_batches, cov_threshold=cov_th
	)

	# 追加: FB ラベルが velocity_mask 内/外にある割合
	cnt_valid = 0
	cnt_in = 0
	cnt_out = 0
	for i, (x, _, _, meta) in enumerate(loader):
		if i >= max_batches:
			break
		fb_idx = meta['fb_idx']  # (B,H), -1=invalid
		offsets = meta['offsets']  # (B,H)
		dt_sec = meta['dt_sec']  # (B,) or (B,1)

		B = fb_idx.size(0)
		W = x.size(-1)  # time samples

		# 可行速度コーン (B,H,W)
		vm = make_velocity_feasible_mask(
			offsets=offsets,
			dt_sec=dt_sec,
			W=W,
			vmin=float(getattr(cfg.loss.fb_seg, 'vmin_mask', 500.0)),
			vmax=float(getattr(cfg.loss.fb_seg, 'vmax_mask', 6000.0)),
			t0_lo_ms=float(getattr(cfg.loss.fb_seg, 't0_lo_ms', -20.0)),
			t0_hi_ms=float(getattr(cfg.loss.fb_seg, 't0_hi_ms', 80.0)),
			taper_ms=float(getattr(cfg.loss.fb_seg, 'taper_ms', 10.0)),
			device=offsets.device,
			dtype=torch.float32,
		)

		valid = fb_idx >= 0
		if valid.any():
			# ラベル位置でのマスク値を抽出
			idx = fb_idx.clamp_min(0).unsqueeze(
				-1
			)  # (B,H,1), invalidは0参照だが後で除外
			m_at = vm.gather(-1, idx).squeeze(-1)  # (B,H) at label time
			m_at = m_at[valid]  # valid only

			cnt_valid += int(valid.sum())
			cnt_in += int((m_at > 0).sum())
			cnt_out += int((m_at <= 0).sum())

	stats['rate_fb_in'] = cnt_in / max(cnt_valid, 1)
	stats['rate_fb_out'] = cnt_out / max(cnt_valid, 1)
	print('dt_sec:', dt_sec)
	print('offsets:', offsets)
	if hasattr(ds, 'close'):
		ds.close()
	return stats


def _audit_for_list(cfg, list_key: str, out_csv: Path | None):
	"""Audit all fields in cfg.<list_key> and optionally write CSV."""
	if not hasattr(cfg, list_key):
		print(f"[INFO] cfg has no '{list_key}', skip.")
		return []

	list_name = getattr(cfg, list_key)
	data_root = Path(cfg.data_root)
	if not data_root.exists():
		print(f'[WARN] data_root not found: {data_root} — skip {list_key}')
		return []

	segy_files, fb_files = _collect_field_files(list_name, str(data_root))
	if not segy_files:
		print(f'[WARN] no files for {list_key}={list_name}')
		return []

	# debug knobs from cfg.debug (with fallbacks)
	audit_batches = int(getattr(getattr(cfg, 'debug', object()), 'audit_batches', 30))
	cov_th = float(getattr(getattr(cfg, 'debug', object()), 'audit_cov_threshold', 0.5))

	rows = []
	for segy, fb in zip(segy_files, fb_files, strict=False):
		field = Path(segy).parent.name
		print(f'\n===== [AUDIT FIELD] {field} ({list_key}) =====')
		print(f'  segy: {segy}')
		print(f'  fb  : {fb}')
		stats = _audit_one_field(
			segy, fb, cfg, max_batches=audit_batches, cov_th=cov_th
		)
		# 簡単な表示を追加
		if 'rate_fb_in' in stats and 'rate_fb_out' in stats:
			print(f'  rate_fb_in:  {stats["rate_fb_in"]:.3f}')
			print(f'  rate_fb_out: {stats["rate_fb_out"]:.3f}')
		row = {'field': field, 'list_key': list_key}
		row.update(stats)
		rows.append(row)

	if out_csv and rows:
		out_csv.parent.mkdir(parents=True, exist_ok=True)
		keys = list(rows[0].keys())
		with out_csv.open('w', newline='') as f:
			w = csv.DictWriter(f, fieldnames=keys)
			w.writeheader()
			for r in rows:
				w.writerow(r)
		print(f'[SUMMARY] wrote {out_csv.resolve()}')

	return rows


def main():
	# hydra config: use relative path from this script → proc/configs
	with initialize(config_path='../configs', version_base='1.3'):
		cfg = compose(config_name='base')

	out_dir = Path('result/audit')
	train_csv = out_dir / 'train_fields.csv'
	valid_csv = out_dir / 'valid_fields.csv'

	_audit_for_list(cfg, 'train_field_list', train_csv)
	_audit_for_list(cfg, 'valid_field_list', valid_csv)

	print('\n[SUMMARY] per-field audits finished.')


if __name__ == '__main__':
	main()

# %%
# %%
# /workspace/proc/debug/vis_fb_mask.py
# %%
# /workspace/proc/debug/vis_fb_mask_multi.py
"""Visualize amplitude + velocity_mask + FB labels
for multiple shots from one field (no section slicing).

- Loads cfg from proc/configs/base.yaml (no argparse)
- Picks one field (by index) from cfg.<which_list>
- Iterates several shots (batch_size=1), computes velocity_mask per shot
- Saves PNGs under result/vis_fb_mask/<field>/shot<k>/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader, SequentialSampler

from proc.util.collate import segy_collate
from proc.util.dataset import MaskedSegyGather
from proc.util.rng_util import worker_init_fn
from proc.util.velocity_mask import make_velocity_feasible_mask

# ===== user knobs =====
which_list = 'train_field_list'  # 'train_field_list' or 'valid_field_list'
field_index = 4  # どのフィールドを可視化するか（0始まり）
max_shots_per_field = 8  # 何ショット表示するか（None で全ショット）
shot_stride = 1  # 例: 2 にすると 1本おきに拾う
n_traces_detail = 8  # 個別表示するトレース本数
# ======================


def _list_path(list_name: str) -> Path:
	here = Path(__file__).resolve()
	cand1 = here.parents[1] / 'configs' / list_name
	cand2 = here.parents[2] / 'configs' / list_name
	if cand1.exists():
		return cand1
	if cand2.exists():
		return cand2
	raise FileNotFoundError(
		f'field list not found: {list_name} (looked in {cand1}, {cand2})'
	)


def _collect_field_files(list_name: str, data_root: Path):
	lp = _list_path(list_name)
	with lp.open() as f:
		fields = [
			ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')
		]
	segy_files, fb_files = [], []
	for field in fields:
		d = data_root / field
		segys = sorted(list(d.glob('*.sgy')) + list(d.glob('*.segy')))
		fbs = sorted(d.glob('*.npy'))
		if not segys or not fbs:
			print(f'[WARN] skip {field}: missing SEG-Y or FB')
			continue
		segy_files.append(segys[0])
		fb_files.append(fbs[0])
	return fields, segy_files, fb_files


@torch.no_grad()
def _shot_vm_and_stats(x, meta, cfg_fb):
	"""Compute velmask for the whole shot and quick stats.
	Returns: vm (1,H,W) float32, stats dict
	"""
	fb_idx = meta['fb_idx']  # (1,H)
	offsets = meta['offsets']  # (1,H)
	dt_sec = meta['dt_sec']  # (1,) or (1,1)
	_, _, H, W = x.shape

	vm = make_velocity_feasible_mask(
		offsets=offsets,
		dt_sec=dt_sec,
		W=W,
		vmin=float(getattr(cfg_fb, 'vmin_mask', 500.0)),
		vmax=float(getattr(cfg_fb, 'vmax_mask', 10000.0)),
		t0_lo_ms=float(getattr(cfg_fb, 't0_lo_ms', -100.0)),
		t0_hi_ms=float(getattr(cfg_fb, 't0_hi_ms', 80.0)),
		taper_ms=float(getattr(cfg_fb, 'taper_ms', 10.0)),
		device=offsets.device,
		dtype=torch.float32,
	)  # (1,H,W)

	coverage = (vm > 0).float().mean(dim=-1)  # (1,H)
	valid = fb_idx >= 0
	rate_fb_in = 0.0
	rate_fb_out = 0.0
	if valid.any():
		idx = fb_idx.clamp_min(0).unsqueeze(-1)  # (1,H,1)
		m_at = vm.gather(-1, idx).squeeze(-1)  # (1,H)
		m_at = m_at[valid]
		rate_fb_in = float((m_at > 0).float().mean().item())
		rate_fb_out = 1.0 - rate_fb_in

	stats = {
		'coverage_med': float(coverage.median().item()),
		'coverage_p05': float(torch.quantile(coverage.flatten(), 0.05).item()),
		'coverage_p95': float(torch.quantile(coverage.flatten(), 0.95).item()),
		'rate_fb_in': rate_fb_in,
		'rate_fb_out': rate_fb_out,
	}
	return vm, stats


def _panel_plot(x, vm, fb_idx, dt_sec, outpng: Path, title: str):
	"""Whole-shot panel: x(1,1,H,W), vm(1,H,W), fb_idx(1,H)"""
	x = x[0, 0].cpu().numpy()
	vm = vm[0].cpu().numpy()
	fb = fb_idx[0].cpu().numpy()
	H, W = x.shape
	dt = float(dt_sec.view(-1)[0].item())
	t_axis = np.arange(W) * dt

	v = np.abs(x)
	vmax = np.quantile(v, 0.995) + 1e-12
	vmin = -vmax

	fig, ax = plt.subplots(figsize=(12, 5.5))
	extent = [t_axis[0], t_axis[-1], 0, H]
	im = ax.imshow(
		x,
		cmap='gray',
		vmin=vmin,
		vmax=vmax,
		extent=extent,
		aspect='auto',
		origin='lower',
		interpolation='nearest',
	)
	ax.imshow(
		vm,
		cmap='Reds',
		alpha=0.25,
		extent=extent,
		aspect='auto',
		origin='lower',
		interpolation='nearest',
	)

	valid = fb >= 0
	if valid.any():
		t_fb = fb[valid] * dt
		h_idx = np.nonzero(valid)[0]
		ax.scatter(
			t_fb, h_idx, s=5, c='cyan', marker='o', linewidths=0.0, label='FB label'
		)

	ax.set_xlabel('Time [s]')
	ax.set_ylabel('Trace index')
	ax.set_title(title)
	if valid.any():
		ax.legend(loc='upper right', fontsize=9)
	cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
	cbar.set_label('Amplitude')
	fig.tight_layout()
	fig.savefig(outpng, dpi=150)
	plt.close(fig)
	print(f'[SAVE] {outpng}')


def _trace_detail_plot(x, vm, fb_idx, dt_sec, outpng: Path, k: int):
	"""Pick several traces across H and plot waveforms + mask + FB."""
	x = x[0, 0].cpu().numpy()  # (H,W)
	vm = vm[0].cpu().numpy()  # (H,W)
	fb = fb_idx[0].cpu().numpy()  # (H,)
	H, W = x.shape
	dt = float(dt_sec.view(-1)[0].item())
	t_axis = np.arange(W) * dt

	if k >= H:
		traces = np.arange(H)
	else:
		traces = np.linspace(0, H - 1, k).round().astype(int)

	fig_h = 2.2 * len(traces)
	fig, axes = plt.subplots(len(traces), 1, figsize=(10, fig_h), sharex=True)
	if len(traces) == 1:
		axes = [axes]

	for ax, h in zip(axes, traces, strict=False):
		sig = x[h]
		sig = sig / (np.std(sig) + 1e-12)
		ax.plot(t_axis, sig, lw=0.7, label=f'trace {h} (norm amp)')
		m = vm[h] > 0
		if m.any():
			ax.fill_between(
				t_axis, -1.2, 1.2, where=m, alpha=0.12, step='pre', label='mask>0'
			)
		if fb[h] >= 0:
			ax.axvline(fb[h] * dt, color='r', ls='--', lw=1.0, label='FB')
		ax.set_ylim(-1.2, 1.2)
		ax.set_ylabel('amp (norm)')
		ax.grid(True, alpha=0.3)
		ax.legend(loc='upper right', fontsize=8)

	axes[-1].set_xlabel('Time [s]')
	fig.tight_layout()
	fig.savefig(outpng, dpi=150)
	plt.close(fig)
	print(f'[SAVE] {outpng}')


def main():
	# load cfg
	with initialize(config_path='../configs', version_base='1.3'):
		cfg = compose(config_name='base')

	data_root = Path(cfg.data_root)
	fields, segys, fbs = _collect_field_files(getattr(cfg, which_list), data_root)
	if not segys:
		print(f'[WARN] no files for list={which_list}')
		return

	# choose one field
	idx = max(0, min(field_index, len(segys) - 1))
	field = fields[idx]
	segy = segys[idx]
	fb = fbs[idx]

	print(f'\n===== [VIS FIELD] {field} ({which_list}) =====')
	print(f'  segy: {segy}')
	print(f'  fb  : {fb}')

	out_root = Path('result/vis_fb_mask') / field
	out_root.mkdir(parents=True, exist_ok=True)

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
		batch_size=1,
		sampler=SequentialSampler(ds),
		shuffle=False,
		num_workers=0,
		pin_memory=True,
		collate_fn=segy_collate,
		drop_last=False,
		worker_init_fn=worker_init_fn,
	)

	shot_count = 0
	for i, (x, _, _, meta) in enumerate(loader):
		if (i % max(1, shot_stride)) != 0:
			continue
		if (
			isinstance(max_shots_per_field, int)
			and max_shots_per_field > 0
			and shot_count >= max_shots_per_field
		):
			break

		vm, stats = _shot_vm_and_stats(x, meta, cfg.loss.fb_seg)
		B, C, H, W = x.shape
		print(
			f'[SHOT {i:04d}] H={H} W={W}  '
			f'coverage med/p05/p95: {stats["coverage_med"]:.3f}/'
			f'{stats["coverage_p05"]:.3f}/{stats["coverage_p95"]:.3f}  '
			f'rate_fb_in/out: {stats["rate_fb_in"]:.3f}/{stats["rate_fb_out"]:.3f}'
		)

		shot_dir = out_root / f'shot{i:04d}'
		shot_dir.mkdir(parents=True, exist_ok=True)

		_panel_plot(
			x,
			vm,
			meta['fb_idx'],
			meta['dt_sec'],
			shot_dir / f'shot{i:04d}_panel.png',
			title=f'{field}  shot {i}',
		)
		_trace_detail_plot(
			x,
			vm,
			meta['fb_idx'],
			meta['dt_sec'],
			shot_dir / f'shot{i:04d}_traces.png',
			k=n_traces_detail,
		)

		shot_count += 1

	if hasattr(ds, 'close'):
		ds.close()
	print(f'\n[DONE] saved {shot_count} shots under: {out_root.resolve()}')


if __name__ == '__main__':
	main()


# %%

import matplotlib.pyplot as plt
import torch


def visualize_pair_quartet(
	noisy: torch.Tensor,  # (1,1,H,W) or (B,1,H,W)
	pred: torch.Tensor,  # (1,1,H,W) or (B,1,H,W)
	clean: torch.Tensor,  # (1,1,H,W) or (B,1,H,W)
	*,
	b: int = 0,
	transpose: bool = True,  # Trueなら(Trace,Time)->(Time,Trace)で表示
	prefix: str = 'pair',
	writer=None,
	global_step: int | None = None,
	amp_range: float = 3.0,  # Noisy/Pred/Clean の表示幅 ±amp_range
	diff_range: float | None = None,  # Diff の表示幅。Noneなら amp_range を流用
):
	# (B,1,H,W) 想定に揃える
	if noisy.dim() == 3:
		noisy = noisy.unsqueeze(0)
	if pred.dim() == 3:
		pred = pred.unsqueeze(0)
	if clean.dim() == 3:
		clean = clean.unsqueeze(0)

	x = noisy[b, 0].detach().cpu().numpy()  # (H,W)
	y = pred[b, 0].detach().cpu().numpy()
	c = clean[b, 0].detach().cpu().numpy()
	d = x - y

	def maybe_T(a):  # (H,W)->(W,H) にしたいとき
		return a.T if transpose else a

	vmin = -amp_range
	vmax = +amp_range
	dmin = -(diff_range if diff_range is not None else amp_range)
	dmax = +(diff_range if diff_range is not None else amp_range)

	fig, axes = plt.subplots(1, 4, figsize=(10, 6), dpi=200, constrained_layout=True)
	axes[0].imshow(
		maybe_T(x),
		aspect='auto',
		vmin=vmin,
		vmax=vmax,
		cmap='seismic',
		interpolation='None',
	)
	axes[0].set_title('Noisy')
	axes[1].imshow(
		maybe_T(y),
		aspect='auto',
		vmin=vmin,
		vmax=vmax,
		cmap='seismic',
		interpolation='None',
	)
	axes[1].set_title('Pred')
	axes[2].imshow(
		maybe_T(c),
		aspect='auto',
		vmin=vmin,
		vmax=vmax,
		cmap='seismic',
		interpolation='None',
	)
	axes[2].set_title('Clean')
	axes[3].imshow(
		maybe_T(d),
		aspect='auto',
		vmin=dmin,
		vmax=dmax,
		cmap='seismic',
		interpolation='None',
	)
	axes[3].set_title('Diff (Noise - Pred)')
	for ax in axes:
		ax.set_xlabel('Time (samples)')
		ax.set_ylabel('Trace')

	if writer is not None:
		writer.add_figure(
			f'{prefix}/synthe_b{b:03d}',
			fig,
			global_step=0 if global_step is None else global_step,
		)

	return fig  # 呼び出し側で plt.close(fig) 推奨


def visualize_recon_triplet(
	x_orig: torch.Tensor,  # (B,1,H,W)
	y_full: torch.Tensor,  # (B,1,H,W)
	fb_idx: torch.Tensor | None = None,  # (B,H) 省略可
	b: int = 0,  # 何番目のサンプルを描くか
	*,
	transpose: bool = True,  # 既存の可視化に合わせて .T にするなら True
	prefix: str = 'val',
	writer=None,
	global_step: int | None = None,
):
	"""- 同一カラースケールで x_orig / y_full を表示
	- fb_idx があれば白点でオーバーレイ（未検出=-1は無視）
	- transpose=True なら (H,W)->(W,H) で表示（あなたの従来スタイル）
	"""
	x = x_orig[b, 0].detach().cpu().numpy()
	y = y_full[b, 0].detach().cpu().numpy()
	d = y - x
	H, W = x.shape

	def _show(ax, img, title):
		im = img.T if transpose else img
		ax.imshow(
			im,
			aspect=1 / 20,
			vmin=-2,
			vmax=2,
			origin='upper',
			cmap='seismic',
			interpolation='None',
		)
		ax.set_title(title)
		ax.set_xlabel('time')
		ax.set_ylabel('trace')
		ax.grid(False)

	fig, axes = plt.subplots(1, 3, figsize=(8, 6), dpi=200, constrained_layout=True)
	_show(axes[0], x, 'Original')
	_show(axes[1], y, 'Reconstruction')
	_show(axes[2], d, 'Difference (y−x)')
	if writer is not None and global_step is not None:
		writer.add_figure(f'{prefix}/field_b{b:03d}', fig, global_step)

	return fig  # その場表示も保存も両方できるように返す

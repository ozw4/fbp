# %%
import numpy as np
import skfmm


def compute_traveltime_3d(v_model, dx, dy, dz, source_xyz, receivers_xyz):
	"""3次元速度モデルに基づく初動走時計算

	Parameters
	----------
	    v_model : np.ndarray, shape (nz, ny, nx)
	        速度モデル [m/s]
	    dx, dy, dz : float
	        セルサイズ [m]
	    source_xyz : (x, y, z)
	        発振位置 [m]
	    receivers_xyz : list of (x, y, z)
	        受振器の位置リスト [m]

	Returns
	-------
	    List of traveltimes [s], 走時フィールド（3D配列）

	"""
	nz, ny, nx = v_model.shape
	slowness = 1.0 / v_model  # スローネス [s/m]

	# Source mask
	phi = np.ones_like(v_model)
	i_src = int(source_xyz[2] / dz)
	j_src = int(source_xyz[1] / dy)
	k_src = int(source_xyz[0] / dx)
	phi[i_src, j_src, k_src] = -1  # source 位置を -1 に

	# Solve 3D eikonal
	tt = skfmm.travel_time(phi, slowness, dx=(dz, dy, dx))  # 注意：Z, Y, X の順

	# Get travel times
	t_arrivals = []
	for x_r, y_r, z_r in receivers_xyz:
		i = int(z_r / dz)
		j = int(y_r / dy)
		k = int(x_r / dx)
		t_arrivals.append(tt[i, j, k])

	return t_arrivals, tt


# 1000×1000×500 m を 10 m グリッドで (nx=100, ny=100, nz=50)
nx = ny = 100
nz = 50
dx = dy = dz = 10.0  # [m]
v = np.full((nz, ny, nx), 2500.0)  # 一様 2500 m/s

# 震源と受振器
src = (500.0, 500.0, 0.0)  # 地表下 50 m
recs = [(x, 400.0, 0.0) for x in np.linspace(100, 900, 100)]

t_rec, tt_vol = compute_traveltime_3d(v, dx, dy, dz, src, recs)

print('first-arrival times [s]:', np.round(t_rec, 3))

import matplotlib.pyplot as plt


def visualize_tt_field(
	tt, dx, dy, dz, src=None, recs=None, slice_type='horizontal', slice_val=0.0
):
	"""Tt         : 3D travel time field (nz, ny, nx)
	dx,dy,dz   : grid spacing [m]
	src        : (x,y,z) or None
	recs       : list of (x,y,z) or None
	slice_type : 'horizontal' | 'inline' | 'crossline'
	slice_val  : depth or position at which to slice [m]
	"""
	nz, ny, nx = tt.shape

	# 軸ラベル
	x = np.arange(nx) * dx
	y = np.arange(ny) * dy
	z = np.arange(nz) * dz

	if slice_type == 'horizontal':
		# z = slice_val で固定した水平断面
		k = int(slice_val / dz)
		plt.imshow(
			tt[k, :, :],
			cmap='hot',
			origin='lower',
			extent=[x.min(), x.max(), y.min(), y.max()],
		)
		plt.xlabel('X [m]')
		plt.ylabel('Y [m]')
		plt.title(f'Travel Time Slice at z={slice_val} m')
		if src:
			plt.plot(src[0], src[1], 'bo', label='Source')
		if recs:
			rx, ry = zip(*[(x, y) for x, y, z in recs], strict=False)
			plt.plot(rx, ry, 'cv', label='Receivers')

	elif slice_type == 'inline':
		# y = slice_val で固定した鉛直断面
		j = int(slice_val / dy)
		plt.imshow(
			tt[:, j, :],
			cmap='hot',
			origin='lower',
			extent=[x.min(), x.max(), z.min(), z.max()],
			aspect='auto',
		)
		plt.xlabel('X [m]')
		plt.ylabel('Z [m]')
		plt.title(f'Inline Slice at y={slice_val} m')
		if src:
			plt.plot(src[0], src[2], 'bo', label='Source')
		if recs:
			rx, rz = zip(
				*[(x, z) for x, y, z in recs if abs(y - slice_val) < dy], strict=False
			)
			plt.plot(rx, rz, 'cv', label='Receivers')

	elif slice_type == 'crossline':
		# x = slice_val で固定した鉛直断面
		i = int(slice_val / dx)
		plt.imshow(
			tt[:, :, i],
			cmap='hot',
			origin='lower',
			extent=[y.min(), y.max(), z.min(), z.max()],
			aspect='auto',
		)
		plt.xlabel('Y [m]')
		plt.ylabel('Z [m]')
		plt.title(f'Crossline Slice at x={slice_val} m')
		if src:
			plt.plot(src[1], src[2], 'bo', label='Source')
		if recs:
			ry, rz = zip(
				*[(y, z) for x, y, z in recs if abs(x - slice_val) < dx], strict=False
			)
			plt.plot(ry, rz, 'cv', label='Receivers')

	else:
		raise ValueError(
			"Invalid slice_type: choose from 'horizontal', 'inline', or 'crossline'"
		)

	plt.colorbar(label='Travel Time [s]')
	if src or recs:
		plt.legend()
	plt.tight_layout()
	plt.show()

	# 水平断面 z=0 で可視化（地表）


visualize_tt_field(
	tt_vol, dx, dy, dz, src=src, recs=recs, slice_type='horizontal', slice_val=0.0
)

# 鉛直断面 y=400m（受振器の直下）
visualize_tt_field(
	tt_vol, dx, dy, dz, src=src, recs=recs, slice_type='inline', slice_val=400.0
)

# 鉛直断面 x=500m（震源直下）
visualize_tt_field(
	tt_vol, dx, dy, dz, src=src, recs=recs, slice_type='crossline', slice_val=500.0
)

# %%

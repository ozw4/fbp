# %%
"""Pure‑Python reimplementation of the RSF `proj` workflow for
Chen et al. (2021) 3D passive seismic source‑location imaging.

It uses the NumPy pseudo‑spectral solver defined in `psm3d_numpy.py`.

Original RSF script sections:
- Part I: parameters
- Part III: build velocity + plot sources
- Part IV: forward modeling to synthesize passive data (+ noise)
- Part V: group receivers and back‑propagate (time‑reversal) per group
- Part VI: cross‑correlation imaging condition to combine groups
- Part VII: (plot results)
- Traditional TRI: single back‑propagation using all receivers

Notes
-----
• Snapshots can be huge. With (81×81×81) and nt=1501, jsnap=4 →
  ~376 snapshots. Each snapshot is ~81^3 ≈ 531,441 doubles (~4.1 MB),
  totaling ~1.5 GB for a single run. Adjust `jsnap` or store reduced
  subsets in practice.
• This script mirrors logic and indices as in the RSF pipeline...
  RSF's `k*=` indices are 1‑based; we convert to 0‑based here.
• For plotting, add matplotlib/numpy I/O as needed.

License: GPLv2+ (inherited from original Chen et al. code)

"""

from __future__ import annotations

import numpy as np

# Import solver pieces from the Python port
from psm3d import ABC, PSMParams, expand_velocity, psm3d

# ----------------------- Part I: Parameters (from RSF) ----------------------
nt = 1501  # number of samples
dt = 0.001  # temporal sampling (s)
nb = 30  # ABC thickness (samples)
ct = 0.01  # ABC decay parameter
jsnap = 4  # snapshot interval (time steps)
ng = 4  # number of receiver groups
nz, nx, ny = 81, 81, 81
dz, dx, dy = 20.0, 20.0, 20.0
ic2 = 1  # 1: product of squares across groups; 0: product then stack

# synthetic test sources
ns = 3
sz = '30,40,50'
sx = '30,40,50'
sy = '30,40,50'
f = '10,10,10'
t = '0.2,0.35,0.5'
A = '1,2,2'

# ---------------- Part III: Velocity model and source positions -------------
# RSF: vel = 1500 + 1.2 * x1, where x1 is physical depth (m)
z_coords = np.arange(nz, dtype=np.float64) * dz
vel_1d = 1500.0 + 1.2 * z_coords  # m/s
override_depth_gradient = True
if override_depth_gradient:
	vel = np.repeat(vel_1d[:, None, None], nx, axis=1)
	vel = np.repeat(vel, ny, axis=2)
else:
	vel = 2000.0 * np.ones((nz, nx, ny), dtype=np.float64)

# Source indices: RSF `k*=` are 1‑based. Convert to 0‑based.
spz = np.array([int(s) - 1 for s in sz.split(',')], dtype=int)
spx = np.array([int(s) - 1 for s in sx.split(',')], dtype=int)
spy = np.array([int(s) - 1 for s in sy.split(',')], dtype=int)
assert len(spz) == ns and len(spx) == ns and len(spy) == ns

f0 = np.array([float(x) for x in f.split(',')], dtype=float)
t0 = np.array([float(x) for x in t.split(',')], dtype=float)
AA = np.array([float(x) for x in A.split(',')], dtype=float)

# ---------------- Part IV: Forward modeling (synthesize data) ---------------
# Geophone layout: RSF example records the whole top plane.
# We place receivers on plane z = gpz at every x,y sample.
# In RSF proj, gplx/gply default to full interior size.

gpz = nb  # after ABC expansion, top interior plane sits at index nb
# For solver params we pass interior sizes; solver will offset internally.
params_fwd = PSMParams(
	nz=nz,
	nx=nx,
	ny=ny,
	dz=dz,
	dx=dx,
	dy=dy,
	nt=nt,
	dt=dt,
	ns=ns,
	spz=spz,
	spx=spx,
	spy=spy,
	f0=f0,
	t0=t0,
	A=AA,
	gpz=0,
	gpx=0,
	gpy=0,
	gplx=nx,
	gply=ny,  # interior indices
	gpz_v=0,
	gpx_v=0,
	gpl_v=-1,  # disabled vertical gather
	jsnap=1,  # RSF data0 asks jsnap=1 to save wfds
	cmplx=True,
	pad1=1,
	ps=True,
	vref=1500.0,
	abc=True,
	abc_params=ABC(
		nbt=nb,
		nbb=nb,
		nblx=nb,
		nbrx=nb,
		nbly=nb,
		nbry=nb,
		ct=ct,
		cb=ct,
		clx=ct,
		crx=ct,
		cly=ct,
		cry=ct,
	),
	verb=False,
)

# Forward model
print(
	'[Forward] Running pseudo‑spectral modeling… (this can be memory intensive if snaps are kept)'
)
dat0, dat_v0, snaps_wfds, _ = psm3d(vel, params_fwd, tri=False)
assert dat0 is not None

# vpad (expanded velocity) for diagnostics (matches RSF `vpad` target)
vpad = expand_velocity(
	vel,
	params_fwd.abc_params.nbt,
	params_fwd.abc_params.nbb,
	params_fwd.abc_params.nblx,
	params_fwd.abc_params.nbrx,
	params_fwd.abc_params.nbly,
	params_fwd.abc_params.nbry,
)

# Add noise like RSF: noise var=1e-7, seed=12005
np.random.seed(12005)
noise_var = 1e-7
if dat0 is not None:
	dat = dat0 + np.sqrt(noise_var) * np.random.standard_normal(dat0.shape)
else:
	dat = None

print(f'data shape: {None if dat is None else dat.shape}')
print(f'wfds (snaps@jsnap=1) shape: {None if snaps_wfds is None else snaps_wfds.shape}')
print(f'vpad shape: {vpad.shape}')

# --------------- Part V: Group receivers & back‑propagate per group ---------
# RSF groups split X into contiguous blocks of width dg = nx//ng

dg = nx // ng
print(f'[Groups] ng={ng}, group width (x) = {dg}')

snaps_groups: list[np.ndarray] = []  # list of arrays shaped (nsnap, nz, nx, ny)
imgs_groups: list[np.ndarray] = []  # final wavefield per group (nz, nx, ny)

# TRI parameters reuse geometry; snapshots saved every `jsnap`
params_tri = PSMParams(
	nz=nz,
	nx=nx,
	ny=ny,
	dz=dz,
	dx=dx,
	dy=dy,
	nt=nt,
	dt=dt,
	ns=0,
	spz=None,
	spx=None,
	spy=None,  # no active source; we inject data
	gpz=0,
	gpx=0,
	gpy=0,
	gplx=nx,
	gply=ny,
	gpz_v=0,
	gpx_v=0,
	gpl_v=-1,
	jsnap=jsnap,
	cmplx=True,
	pad1=1,
	ps=True,
	vref=1500.0,
	abc=True,
	abc_params=ABC(
		nbt=nb,
		nbb=nb,
		nblx=nb,
		nbrx=nb,
		nbly=nb,
		nbry=nb,
		ct=ct,
		cb=ct,
		clx=ct,
		crx=ct,
		cly=ct,
		cry=ct,
	),
	verb=False,
)

for gi in range(ng):
	x0 = gi * dg
	x1 = x0 + dg
	# Build masked data: keep x in [x0:x1), all y; zero elsewhere
	dat_mask = np.zeros_like(dat)
	dat_mask[:, x0:x1, :] = dat[:, x0:x1, :]
	print(f'[TRI group {gi}] Injecting x∈[{x0}:{x1})')

	dat_back, dat_v_back, snaps_g, img_g = psm3d(
		vel, params_tri, tri=True, dat_in=dat_mask, dat_v_in=None
	)
	assert snaps_g is not None and img_g is not None
	snaps_groups.append(snaps_g)
	imgs_groups.append(img_g)

print('[TRI groups] collected snapshots per group:', [s.shape for s in snaps_groups])

# --------------- Part VI: Cross‑correlation imaging condition ---------------
# RSF ic2: product of squares across groups, then stack over time.
# snaps_groups list elements are shaped (nsnap, nz, nx, ny)

if ic2:
	# elementwise ∏_g (snaps_g^2)
	prod_sq = None
	for g, sg in enumerate(snaps_groups):
		term = sg * sg
		prod_sq = term if prod_sq is None else (prod_sq * term)
	ccr0 = prod_sq
else:
	# alternative: concatenate along a new axis then take product over that axis
	stacked = np.stack(snaps_groups, axis=0)  # (ng, nsnap, nz, nx, ny)
	ccr0 = np.prod(stacked, axis=0)

# location image: sum over time snapshots
location0 = np.sum(ccr0, axis=0)  # (nz, nx, ny)

# --------------- Traditional Time‑Reversal Imaging (single pass) ------------
# Inject full unmasked data once; take energy snapshots and optionally pick frames
print('[TRI full] Back‑propagating full data once…')
_, _, snaps_full, _ = psm3d(vel, params_tri, tri=True, dat_in=dat, dat_v_in=None)
assert snaps_full is not None
snaps_abs0 = snaps_full * snaps_full

# Example frames roughly matching RSF windows (indices will differ if jsnap≠1)
# Here jsnap=4, so RSF f4=50 → index 50 in snaps array.
frames_to_view = [50, 83, 130]
frames_to_view = [i for i in frames_to_view if i < snaps_abs0.shape[0]]
location_tr_frames = [snaps_abs0[i] for i in frames_to_view]  # list of (nz,nx,ny)

# Summed energy image as a simple alternative
location_tr_sum = np.sum(snaps_abs0, axis=0)

# ----------------------------- Outputs / Saving -----------------------------
# Replace with np.save or visualization as needed. Here we just report shapes.
print('location0 (grouped CC) shape:', location0.shape)
print('TR frames picked:', [f for f in frames_to_view])
print('TR sum image shape:', location_tr_sum.shape)

# Example saves (commented out):
# np.save('vel.npy', vel)
# np.save('vpad.npy', vpad)
# np.save('data.npy', dat)
# for i, sg in enumerate(snaps_groups):
#     np.save(f'snaps_group{i}.npy', sg)
# np.save('location0.npy', location0)
# for i, fr in enumerate(location_tr_frames):
#     np.save(f'location_tr_frame{i}.npy', fr)
# np.save('location_tr_sum.npy', location_tr_sum)

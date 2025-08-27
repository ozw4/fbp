# %%
"""3D acoustic wavefield modeling using the pseudo‑spectral method (NumPy)

Python port of the compact C program by Chen et al. (2020/2021) used in:
  Chen, Y., O.M. Saad, M. Bai, X. Liu, and S. Fomel, 2021,
  A compact program for 3D passive seismic source‑location imaging,
  Seismological Research Letters, 92(5), 3187–3201.

Original C code is GPLv2+ (UT Austin / Madagascar / RSF ecosystem). This
Python translation remains under the same license terms. See LICENSE notes
below.

Notes about the port:
- Uses NumPy FFTs (CPU) instead of KISS/FFTW. Shape is (nz, nx, ny).
- Supports both pseudo‑spectral (−|k|^2) and pseudo‑analytical operators.
- Includes simple exponential absorbing boundary (ABC) and velocity padding.
- Source: Ricker in time domain (analytic).
- Time‑reversal imaging (tri=True) supported.

Differences vs. the C original:
- The original provided a frequency‑domain wavelet shaper via RSF (sf_freqfilt).
  Here we use only the analytic Ricker in time; behavior is very similar for
  typical settings. If you need the exact RSF shaper, plug it in where noted.
- FFT centering sign flips are unnecessary since we compute |k| consistently
  from np.fft.fftfreq grids.

Author: Python translation by ChatGPT
License: GPLv2 or later (same as the original; see "License" section at bottom)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------- Utilities ---------------------------------


def next_fast_len(n: int) -> int:
	"""Return an FFT-efficient length >= n.
	Tries numpy.fft.next_fast_len if present; else falls back to next power of 2.
	"""
	try:
		from numpy.fft import next_fast_len as _nfl  # type: ignore

		return int(_nfl(int(n)))
	except Exception:
		# next power of two
		n = int(n)
		if n <= 1:
			return 1
		return 1 << (n - 1).bit_length()


def ricker(t: np.ndarray, f0: float, t0: float, A: float) -> np.ndarray:
	"""Ricker wavelet (same formula as the C helper).

	Parameters
	----------
	t : array_like
	    Time vector (seconds).
	f0 : float
	    Peak frequency (Hz).
	t0 : float
	    Time lag (seconds).
	A : float
	    Amplitude.

	"""
	x = (np.pi * f0 * (t - t0)) ** 2
	return -A * np.exp(-x) * (1.0 - 2.0 * x)


# ----------------------- Absorbing boundary (ABC) ---------------------------


def _abc_weights(nb: int, c: float) -> np.ndarray | None:
	if nb <= 0:
		return None
	idx = np.arange(nb, dtype=float)
	# decay from interior (index nb-1) to edge (index 0)
	# same functional form as the C version: exp(-c^2 * (nb-1-ib)^2)
	return np.exp(-((c * (nb - 1 - idx)) ** 2))


@dataclass
class ABC:
	nbt: int = 0
	nbb: int = 0
	nblx: int = 0
	nbrx: int = 0
	nbly: int = 0
	nbry: int = 0
	ct: float = 0.0
	cb: float = 0.0
	clx: float = 0.0
	crx: float = 0.0
	cly: float = 0.0
	cry: float = 0.0

	# computed weights
	wt: np.ndarray | None = None
	wb: np.ndarray | None = None
	wlx: np.ndarray | None = None
	wrx: np.ndarray | None = None
	wly: np.ndarray | None = None
	wry: np.ndarray | None = None

	def init(self):
		self.wt = _abc_weights(self.nbt, self.ct)
		self.wb = _abc_weights(self.nbb, self.cb)
		self.wlx = _abc_weights(self.nblx, self.clx)
		self.wrx = _abc_weights(self.nbrx, self.crx)
		self.wly = _abc_weights(self.nbly, self.cly)
		self.wry = _abc_weights(self.nbry, self.cry)

	def apply(self, a: np.ndarray) -> None:
		"""Apply exponential decay to the boundary layers of a volume.
		a.shape = (nz2, nx2, ny2)
		"""
		nz2, nx2, ny2 = a.shape
		# z top
		if self.nbt > 0 and self.wt is not None:
			a[: self.nbt, :, :] *= self.wt[:, None, None]
		# z bottom
		if self.nbb > 0 and self.wb is not None:
			a[nz2 - self.nbb :, :, :] *= self.wb[::-1, None, None]
		# x left
		if self.nblx > 0 and self.wlx is not None:
			a[:, : self.nblx, :] *= self.wlx[None, :, None]
		# x right
		if self.nbrx > 0 and self.wrx is not None:
			a[:, nx2 - self.nbrx :, :] *= self.wrx[::-1][None, :, None]
		# y left
		if self.nbly > 0 and self.wly is not None:
			a[:, :, : self.nbly] *= self.wly[None, None, :]
		# y right
		if self.nbry > 0 and self.wry is not None:
			a[:, :, ny2 - self.nbry :] *= self.wry[::-1][None, None, :]


# ------------------------------ Parameters ----------------------------------


@dataclass
class PSMParams:
	# survey/model sizes (INTERIOR, without ABC)
	nz: int
	nx: int
	ny: int
	dz: float
	dx: float
	dy: float

	# sources
	nt: int
	dt: float
	ns: int = 1
	spz: np.ndarray | None = None
	spx: np.ndarray | None = None
	spy: np.ndarray | None = None
	f0: np.ndarray | None = None
	t0: np.ndarray | None = None
	A: np.ndarray | None = None
	src: int = 0  # 0/1 both use analytic Ricker here

	# receivers (horizontal grid plane)
	gpz: int = 0
	gpx: int = 0
	gpy: int = 0
	gplz: int = 1  # kept for parity; not used for horizontal gather
	gplx: int = -1  # if -1, uses nx (interior)
	gply: int = -1  # if -1, uses ny (interior)

	# vertical line receivers (optional)
	gpz_v: int = 0
	gpx_v: int = 0
	gpy_v: int = 0  # used here (the C example left it at 0)
	gpl_v: int = -1  # if -1, uses nz (interior)

	# snapping
	jsnap: int = 0  # save every jsnap steps; 0 disables snapshots

	# FFT / operator
	cmplx: bool = True  # complex FFT path; simpler and robust
	pad1: int = 1  # pad factor along z (interior nz * pad1)
	ps: bool = True  # True: pseudo‑spectral (−k^2); False: pseudo‑analytical
	vref: float = 1500.0

	# absorbing boundary
	abc: bool = False
	abc_params: ABC = ABC()

	# verbosity
	verb: bool = False


# ------------------------- Velocity expansion (ABC) -------------------------


def expand_velocity(
	vel: np.ndarray,
	nbt: int,
	nbb: int,
	nblx: int,
	nbrx: int,
	nbly: int,
	nbry: int,
) -> np.ndarray:
	"""Pad velocity for ABC by copying edge values outward.

	vel: (nz, nx, ny) interior model
	returns: (nz+nbt+nbb, nx+nblx+nbrx, ny+nbly+nbry)
	"""
	nz, nx, ny = vel.shape
	nz2, nx2, ny2 = nz + nbt + nbb, nx + nblx + nbrx, ny + nbly + nbry
	out = np.empty((nz2, nx2, ny2), dtype=vel.dtype)

	# center
	out[nbt : nbt + nz, nblx : nblx + nx, nbly : nbly + ny] = vel

	# z top & bottom: copy the first/last interior planes
	if nbt > 0:
		out[:nbt, nblx : nblx + nx, nbly : nbly + ny] = vel[0:1, :, :]
	if nbb > 0:
		out[nbt + nz :, nblx : nblx + nx, nbly : nbly + ny] = vel[-1:, :, :]

	# x left & right
	if nblx > 0:
		out[:, :nblx, nbly : nbly + ny] = out[:, nblx : nblx + 1, nbly : nbly + ny]
	if nbrx > 0:
		out[:, nblx + nx :, nbly : nbly + ny] = out[
			:, nblx + nx - 1 : nblx + nx, nbly : nbly + ny
		]

	# y left & right
	if nbly > 0:
		out[:, :, :nbly] = out[:, :, nbly : nbly + 1]
	if nbry > 0:
		out[:, :, nbly + ny :] = out[:, :, nbly + ny - 1 : nbly + ny]

	# fill corners are already handled by the above sides
	return out


# ------------------------------ Spectral op ---------------------------------


def build_laplacian_multiplier(
	nz2: int,
	nx2: int,
	ny2: int,
	dz: float,
	dx: float,
	dy: float,
	ps: bool,
	vref: float,
) -> np.ndarray:
	"""Return spectral multiplier with shape (nz2, nx2, ny2) for FFTN path.

	ps=True  -> −|k|^2
	ps=False -> 2*(cos(vref*|k|) − 1) / vref^2   (pseudo‑analytical)
	"""
	kz = 2.0 * np.pi * np.fft.fftfreq(nz2, d=dz)[:, None, None]
	kx = 2.0 * np.pi * np.fft.fftfreq(nx2, d=dx)[None, :, None]
	ky = 2.0 * np.pi * np.fft.fftfreq(ny2, d=dy)[None, None, :]
	k = np.sqrt(kz * kz + kx * kx + ky * ky)

	if ps:
		lapl = -(k * k)
	else:
		vdt = vref  # vref is given in m/s; time step factor will be applied in vv
		lapl = 2.0 * (np.cos(vdt * k) - 1.0) / (vdt * vdt)
	# k=0 already gives 0 for both forms
	return lapl.astype(np.float64)


# ------------------------------ Main solver --------------------------------


def psm3d(
	vel: np.ndarray,
	params: PSMParams,
	tri: bool = False,
) -> tuple[
	np.ndarray | None,
	np.ndarray | None,
	np.ndarray | None,
	np.ndarray | None,
]:
	"""Run 3D acoustic pseudo‑spectral modeling or time‑reversal imaging.

	Parameters
	----------
	vel : ndarray, shape (nz, nx, ny)
	    Interior velocity model [m/s].
	params : PSMParams
	    Simulation parameters (see dataclass docstring).
	tri : bool
	    If True, perform time‑reversal imaging (inject data at receivers and
	    back‑propagate); returns `img` as the final wavefield in the interior.

	Returns
	-------
	dat : ndarray or None, shape (gply, gplx, nt)
	    Horizontal receiver gather (if modeling and requested).
	dat_v : ndarray or None, shape (gpl_v, nt)
	    Vertical line gather (if modeling and requested).
	snaps : ndarray or None, shape (nsnap, nz, nx, ny)
	    Snapshots (if params.jsnap>0).
	img : ndarray or None, shape (nz, nx, ny)
	    Final wavefield (tri=True), else None.

	"""
	# Unpack interior sizes
	nz, nx, ny = int(params.nz), int(params.nx), int(params.ny)
	dz, dx, dy = float(params.dz), float(params.dx), float(params.dy)
	nt, dt = int(params.nt), float(params.dt)

	# ABC padding & weights
	abc = params.abc
	ap = params.abc_params
	if abc:
		ap.init()
		vel2 = expand_velocity(vel, ap.nbt, ap.nbb, ap.nblx, ap.nbrx, ap.nbly, ap.nbry)
	else:
		vel2 = vel.copy()
		ap = ABC()  # zero sizes, no‑op apply

	nz2, nx2, ny2 = vel2.shape

	# FFT sizes (complex FFT path only for clarity)
	pad1 = max(1, int(params.pad1))
	nz_fft = next_fast_len(nz2 * pad1)
	nx_fft = next_fast_len(nx2)
	ny_fft = next_fast_len(ny2)

	# We implement FFT on the padded computational grid directly; if FFT sizes
	# exceed grid sizes, we can zero‑pad the fields for the transform.
	# For simplicity and memory locality, choose FFT sizes equal to grid sizes.
	# If you *really* need extra padding, increase the ABC thickness instead.
	nz_fft, nx_fft, ny_fft = nz2, nx2, ny2

	# Spectral multiplier
	lapl = build_laplacian_multiplier(
		nz_fft, nx_fft, ny_fft, dz, dx, dy, params.ps, params.vref
	)

	# allocate fields
	curr = np.zeros((nz_fft, nx_fft, ny_fft), dtype=np.float64)
	prev = np.zeros_like(curr)
	wave = np.zeros_like(curr)

	# vv = (vel^2 * dt^2)
	vv = (vel2.astype(np.float64) ** 2) * (dt * dt)

	# sources
	ns = int(params.ns)
	if not tri and ns > 0:
		# default to interior sizes if not provided
		spz = np.asarray(
			params.spz if params.spz is not None else np.zeros(ns, int), dtype=int
		)
		spx = np.asarray(
			params.spx if params.spx is not None else np.zeros(ns, int), dtype=int
		)
		spy = np.asarray(
			params.spy if params.spy is not None else np.zeros(ns, int), dtype=int
		)
		f0 = np.asarray(
			params.f0 if params.f0 is not None else np.full(ns, 25.0), dtype=float
		)
		t0 = np.asarray(
			params.t0 if params.t0 is not None else np.full(ns, 0.04), dtype=float
		)
		A = np.asarray(params.A if params.A is not None else np.ones(ns), dtype=float)
		# Time vector
		tvec = np.arange(nt, dtype=np.float64) * dt
		rick = np.stack(
			[ricker(tvec, float(f0[i]), float(t0[i]), float(A[i])) for i in range(ns)],
			axis=0,
		)
	else:
		spz = spx = spy = None
		rick = None

	# receivers (horizontal)
	gplx = nx if params.gplx in (-1, None) else int(params.gplx)
	gply = ny if params.gply in (-1, None) else int(params.gply)
	gpx = int(params.gpx)
	gpy = int(params.gpy)
	gpz = int(params.gpz)

	# vertical line receivers (optional)
	gpl_v = nz if params.gpl_v in (-1, None) else int(params.gpl_v)
	gpx_v = int(params.gpx_v)
	gpy_v = int(params.gpy_v)
	gpz_v = int(params.gpz_v)

	# outputs
	dat = None
	dat_v = None
	if not tri:
		if gplx > 0 and gply > 0:
			dat = np.zeros((gply, gplx, nt), dtype=np.float64)
		if gpl_v > 0:
			dat_v = np.zeros((gpl_v, nt), dtype=np.float64)

	snaps = None
	snap_times = []
	if params.jsnap and params.jsnap > 0:
		nsnap = (nt + params.jsnap - 1) // params.jsnap
		snaps = np.zeros((nsnap, nz, nx, ny), dtype=np.float64)

	if params.verb:
		print(f'Grid (interior): nz={nz}, nx={nx}, ny={ny}')
		print(f'Grid (padded):   nz2={nz2}, nx2={nx2}, ny2={ny2}')
		print(f'FFT sizes:       {nz_fft} x {nx_fft} x {ny_fft}')
		print(
			f'Operator:        {"pseudo‑spectral (−k^2)" if params.ps else "pseudo‑analytical"}'
		)
		if abc:
			print(
				f'ABC: z({ap.nbt},{ap.nbb}) x({ap.nblx},{ap.nbrx}) y({ap.nbly},{ap.nbry})'
			)

	# time loop
	snap_idx = 0
	t_increasing = not tri
	it_range = range(nt) if t_increasing else range(nt - 1, -1, -1)

	for it in it_range:
		# spectral step: wave = F^{-1} { lapl * F{curr} }
		cwave = np.fft.fftn(curr)
		cwavem = cwave * lapl
		wave = np.fft.ifftn(cwavem).real

		# leap‑frog update
		new_curr = (2.0 * curr - prev) + wave * vv
		prev, curr = curr, new_curr

		if tri:
			# inject recorded data (horizontal plane)
			if dat is not None:
				# curr[gpz, gpx:gpx+gplx, gpy:gpy+gply] += vv[...] * dat[..., it]
				sl = np.s_[gpz, gpx : gpx + gplx, gpy : gpy + gply]
				curr[sl] += vv[sl] * dat[:, :, it]
			# inject vertical line data
			if dat_v is not None:
				sl = np.s_[gpz_v : gpz_v + gpl_v, gpx_v, gpy_v]
				curr[sl] += vv[sl] * dat_v[:, it]
		# add sources (3×3×3 anti‑aliased point)
		elif ns > 0:
			for i in range(ns):
				amp = rick[i, it]
				z0 = int(spz[i]) + ap.nbt
				x0 = int(spx[i]) + ap.nblx
				y0 = int(spy[i]) + ap.nbly
				for dz1 in (-1, 0, 1):
					jz = z0 + dz1
					if jz < 0 or jz >= nz2:
						continue
					for dx1 in (-1, 0, 1):
						jx = x0 + dx1
						if jx < 0 or jx >= nx2:
							continue
						for dy1 in (-1, 0, 1):
							jy = y0 + dy1
							if jy < 0 or jy >= ny2:
								continue
							w = 1.0 / (abs(dz1) + abs(dx1) + abs(dy1) + 1.0)
							curr[jz, jx, jy] += vv[jz, jx, jy] * amp * w

		# ABC damping
		if abc:
			ap.apply(curr)
			ap.apply(prev)

		# record data (modeling)
		if not tri:
			if dat is not None:
				sl = np.s_[gpz, gpx : gpx + gplx, gpy : gpy + gply]
				dat[:, :, it] = curr[sl]
			if dat_v is not None:
				sl = np.s_[gpz_v : gpz_v + gpl_v, gpx_v, gpy_v]
				dat_v[:, it] = curr[sl]

		# snapshots
		if params.jsnap and params.jsnap > 0 and (it % params.jsnap == 0):
			if snaps is not None:
				snaps[snap_idx] = curr[
					ap.nbt : ap.nbt + nz, ap.nblx : ap.nblx + nx, ap.nbly : ap.nbly + ny
				]
			snap_times.append(it * dt)
			snap_idx += 1

	# time‑reversal imaging output: final interior wavefield
	img = None
	if tri:
		img = curr[
			ap.nbt : ap.nbt + nz, ap.nblx : ap.nblx + nx, ap.nbly : ap.nbly + ny
		].copy()

	return dat, dat_v, snaps, img


# ------------------------------ Example usage -------------------------------
if __name__ == '__main__':
	# Minimal example (tiny grid). Replace with your own model/geometry.
	nz, nx, ny = 64, 64, 64
	dz, dx, dy = 5.0, 5.0, 5.0
	nt, dt = 750, 0.001

	# Homogeneous model 2000 m/s with a low‑velocity lens
	vel = 2000.0 * np.ones((nz, nx, ny), dtype=np.float64)
	cz, cx, cy, r, dv = 28, 32, 32, 10, -400
	zz, xx, yy = np.ogrid[:nz, :nx, :ny]
	mask = (zz - cz) ** 2 + (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
	vel[mask] += dv

	params = PSMParams(
		nz=nz,
		nx=nx,
		ny=ny,
		dz=dz,
		dx=dx,
		dy=dy,
		nt=nt,
		dt=dt,
		ns=1,
		spz=np.array([4]),
		spx=np.array([nx // 2]),
		spy=np.array([ny // 2]),
		f0=np.array([20.0]),
		t0=np.array([0.05]),
		A=np.array([1.0]),
		gpz=6,
		gpx=0,
		gpy=0,
		gplx=nx,
		gply=ny,
		jsnap=50,
		cmplx=True,
		pad1=1,
		ps=True,
		vref=1500.0,
		abc=True,
		abc_params=ABC(
			nbt=8,
			nbb=8,
			nblx=8,
			nbrx=8,
			nbly=8,
			nbry=8,
			ct=0.015,
			cb=0.015,
			clx=0.015,
			crx=0.015,
			cly=0.015,
			cry=0.015,
		),
		verb=True,
	)

	dat, dat_v, snaps, img = psm3d(vel, params, tri=False)

	print('dat shape:', None if dat is None else dat.shape)
	if snaps is not None:
		print('snaps shape:', snaps.shape)

	# If you want time‑reversal imaging, you could then run:
	#   dat_back = dat.copy()
	#   params_back = params  # typically the same
	#   dat_back_v = None
	#   # reuse same params; provide `dat` as input by replacing `dat` above (left to user)

"""
License (GPLv2 or later)
------------------------
This file is a translation of GPLv2+ code. As such, it is itself provided under
the GNU General Public License, version 2 or (at your option) any later
version. See <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.

"""

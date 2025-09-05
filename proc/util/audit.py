# proc/util/audit.py
import math
from typing import Any

import torch

from proc.util.velocity_mask import make_velocity_feasible_mask


@torch.no_grad()
def assert_meta_shapes(meta: dict, B: int, H: int):
    """Lightweight shape/finite checks for meta fields used by velocity mask."""
    offs = meta['offsets']
    dt = meta['dt_sec']
    assert offs.ndim == 2 and offs.shape == (B, H), f"offsets shape {offs.shape}, expected {(B,H)}"
    assert torch.isfinite(offs).all(), "offsets has NaN/Inf"
    assert dt.ndim in (1, 2) and dt.shape[0] == B, f"dt_sec shape {dt.shape}, expected (B,) or (B,1)"
    assert torch.isfinite(dt).all(), "dt_sec has NaN/Inf"

@torch.no_grad()
def audit_offsets_and_mask_coverage(
    loader,
    cfg_fb,
    max_batches: int = 100,
    cov_threshold: float = 0.5,
) -> dict[str, Any]:
    """Scan a few batches and report:
    - dx stats (min/median/max), count of dx==0
    - traces where velocity mask is all-zero
    - fraction of target probability mass that lies inside the velocity cone
    - crude checks for offsets sanity
    """
    dx0_total = 0
    zero_mask_tr = 0
    low_cov_tr = 0
    total_tr = 0

    nan_off_elems = 0
    nan_dt_elems = 0
    all_equal_shot = 0  # offsets constant over H (likely unset)
    nonmonotonic_shot = 0

    dx_min = float("inf")
    dx_max = 0.0
    dx_meds = []

    for bi, (x, target, _, meta) in enumerate(loader):
        B, _, H, W = x.shape
        offs = meta['offsets']
        dt = meta['dt_sec']
        fb_idx = meta['fb_idx']
        total_tr += B * H

        # Shapes & finite checks
        try:
            assert_meta_shapes(meta, B, H)
        except AssertionError as e:
            print(f"[AUDIT][batch {bi}] SHAPE/FIN ERROR: {e}")
            # continue; still try to collect more info

        # NaN/Inf counters (element-wise)
        nan_off_elems += int((~torch.isfinite(offs)).sum().item())
        nan_dt_elems  += int((~torch.isfinite(dt)).sum().item())

        # Per-shot "all offsets equal?" and monotonicity probes
        # all_equal over H (no variation → likely missing offsets)
        diffs = (offs[:, 1:] - offs[:, :-1])
        all_equal_shot += int((diffs.abs().sum(dim=1) == 0).sum().item())
        nonmonotonic_shot += int(((diffs >= 0).all(dim=1) | (diffs <= 0).all(dim=1)).logical_not().sum().item())

        # dx stats over H
        dx = diffs.abs()
        dx0_total += int((dx == 0).sum().item())
        if dx.numel() > 0:
            dx_min = min(dx_min, float(dx.min().cpu()))
            dx_max = max(dx_max, float(dx.max().cpu()))
            dx_meds.append(float(dx.median().cpu()))

        # Velocity-cone mask
        velmask = make_velocity_feasible_mask(
            offsets=offs,
            dt_sec=dt,
            W=W,
            vmin=float(getattr(cfg_fb, 'vmin_mask', 500.0)),
            vmax=float(getattr(cfg_fb, 'vmax_mask', 6000.0)),
            t0_lo_ms=float(getattr(cfg_fb, 't0_lo_ms', -20.0)),
            t0_hi_ms=float(getattr(cfg_fb, 't0_hi_ms', 80.0)),
            taper_ms=float(getattr(cfg_fb, 'taper_ms', 10.0)),
            device=offs.device,
            dtype=offs.dtype,
        )

        # Detect traces whose mask is entirely zero
        zero_mask = (velmask.sum(dim=-1) == 0)  # (B,H)
        zero_mask_tr += int(zero_mask.sum().item())

        # Target coverage inside the cone (per trace)
        q = target.squeeze(1).to(velmask)                  # (B,H,W)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        cov = (q * velmask).sum(dim=-1)                    # (B,H) in [0,1]

        valid = (fb_idx >= 0)
        low_cov = (cov < cov_threshold) & valid
        low_cov_tr += int(low_cov.sum().item())

        if (bi + 1) >= max_batches:
            break

    dx_med = float(torch.tensor(dx_meds).median().item()) if dx_meds else float("nan")

    stats = {
        "traces": int(total_tr),
        "dx_min": dx_min if math.isfinite(dx_min) else float("nan"),
        "dx_med": dx_med,
        "dx_max": dx_max if math.isfinite(dx_max) else float("nan"),
        "dx_eq0": int(dx0_total),
        "mask_all_zero_traces": int(zero_mask_tr),
        "low_coverage_traces": int(low_cov_tr),
        "nan_off_elems": int(nan_off_elems),
        "nan_dt_elems": int(nan_dt_elems),
        "all_equal_shots": int(all_equal_shot),
        "nonmonotonic_shots": int(nonmonotonic_shot),
        "cov_threshold": float(cov_threshold),
    }

    # Pretty print summary
    print("[AUDIT] traces:            ", stats["traces"])
    print(f"[AUDIT] dx[m] min/med/max: {stats['dx_min']:.6g} / {stats['dx_med']:.6g} / {stats['dx_max']:.6g}")
    print( "[AUDIT] dx==0 count:       ", stats["dx_eq0"])
    print( "[AUDIT] mask==0 traces:    ", stats["mask_all_zero_traces"])
    print( "[AUDIT] low coverage (<{:.2f}) valid traces: {}".format(stats["cov_threshold"], stats["low_coverage_traces"]))
    print( "[AUDIT] NaN/Inf offsets elts:", stats["nan_off_elems"], "  NaN/Inf dt_sec elts:", stats["nan_dt_elems"])
    print( "[AUDIT] shots all-equal offsets:", stats["all_equal_shots"],
           "  non-monotonic shots (mixed up/down):", stats["nonmonotonic_shots"])

    return stats

"""Helper script to compute robust normalization statistics for training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from hydra import compose, initialize

from proc.util.dataset import MaskedSegyGather


def _resolve_field_list_path(list_name: str) -> Path:
    """Return the path to the field list file, searching common locations."""

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / 'configs' / list_name,
        here.parents[2] / 'configs' / list_name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f'Field list not found for {list_name!r}; looked in: '
        f"{', '.join(str(c) for c in candidates)}"
    )


def _collect_field_files(list_name: str, data_root: str) -> tuple[list[Path], list[Path]]:
    """Collect SEG-Y and label file pairs listed in ``list_name``."""

    list_path = _resolve_field_list_path(list_name)
    with list_path.open() as f:
        fields = [
            ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')
        ]

    segy_files: list[Path] = []
    fb_files: list[Path] = []
    for field in fields:
        base = Path(data_root) / field
        segy_candidates = sorted(list(base.glob('*.sgy')) + list(base.glob('*.segy')))
        fb_candidates = sorted(base.glob('*.npy'))
        if not segy_candidates or not fb_candidates:
            print(f'[WARN] Skipping {field}: missing SEG-Y or FB files')
            continue
        segy_files.append(segy_candidates[0])
        fb_files.append(fb_candidates[0])

    if not segy_files:
        raise RuntimeError(f'No usable SEG-Y files found for list {list_name!r}')

    return segy_files, fb_files


def _build_train_dataset(cfg, data_root: str, field_list: str) -> MaskedSegyGather:
    """Instantiate the training dataset with the same configuration as training."""

    segy_files, fb_files = _collect_field_files(field_list, data_root)

    dataset_cfg = getattr(cfg, 'dataset', object())
    augment_cfg = getattr(dataset_cfg, 'augment', None)
    time_cfg = getattr(augment_cfg, 'time', None) if augment_cfg is not None else None
    space_cfg = getattr(augment_cfg, 'space', None) if augment_cfg is not None else None
    freq_cfg = getattr(augment_cfg, 'freq', None) if augment_cfg is not None else None
    return MaskedSegyGather(
        [str(p) for p in segy_files],
        [str(p) for p in fb_files],
        primary_keys=getattr(dataset_cfg, 'primary_keys', None),
        primary_key_weights=getattr(dataset_cfg, 'primary_key_weights', None),
        use_header_cache=getattr(dataset_cfg, 'use_header_cache', False),
        header_cache_dir=getattr(dataset_cfg, 'header_cache_dir', None),
        use_superwindow=getattr(dataset_cfg, 'use_superwindow', False),
        sw_halfspan=getattr(dataset_cfg, 'sw_halfspan', 0),
        sw_prob=getattr(dataset_cfg, 'sw_prob', 0.3),
        mask_ratio=getattr(dataset_cfg, 'mask_ratio', 0.0),
        mask_mode=getattr(dataset_cfg, 'mask_mode', 'replace'),
        mask_noise_std=getattr(dataset_cfg, 'mask_noise_std', 1.0),
        target_mode=getattr(dataset_cfg, 'target_mode', 'recon'),
        label_sigma=getattr(dataset_cfg, 'label_sigma', 1.0),
        flip=getattr(dataset_cfg, 'flip', False),
        augment_time_prob=getattr(time_cfg, 'prob', 0.0),
        augment_time_range=tuple(getattr(time_cfg, 'range', (0.95, 1.05)))
        if time_cfg is not None
        else (0.95, 1.05),
        augment_space_prob=getattr(space_cfg, 'prob', 0.0),
        augment_space_range=tuple(getattr(space_cfg, 'range', (0.90, 1.10)))
        if space_cfg is not None
        else (0.90, 1.10),
        augment_freq_prob=getattr(freq_cfg, 'prob', 0.0),
        augment_freq_kinds=tuple(getattr(freq_cfg, 'kinds', ()))
        if freq_cfg is not None
        else tuple(),
        augment_freq_band=tuple(getattr(freq_cfg, 'band', (0.05, 0.45)))
        if freq_cfg is not None
        else (0.05, 0.45),
        augment_freq_width=tuple(getattr(freq_cfg, 'width', (0.10, 0.35)))
        if freq_cfg is not None
        else (0.10, 0.35),
        augment_freq_roll=getattr(freq_cfg, 'roll', 0.02) if freq_cfg is not None else 0.02,
        augment_freq_restandardize=getattr(freq_cfg, 'restandardize', True)
        if freq_cfg is not None
        else True,
        reject_fblc=getattr(dataset_cfg, 'reject_fblc', False),
        fblc_percentile=getattr(dataset_cfg, 'fblc_percentile', 95.0),
        fblc_thresh_ms=getattr(dataset_cfg, 'fblc_thresh_ms', 8.0),
        fblc_min_pairs=getattr(dataset_cfg, 'fblc_min_pairs', 16),
        fblc_apply_on=getattr(dataset_cfg, 'fblc_apply_on', 'any'),
        valid=False,
    )


def compute_norm_stats(
    cfg,
    *,
    data_root: str,
    train_field_list: str,
) -> tuple[float, float]:
    """Compute the 95th percentile statistics for offsets and record lengths."""

    dataset = _build_train_dataset(cfg, data_root=data_root, field_list=train_field_list)
    try:
        all_offsets: list[np.ndarray] = []
        all_tend_ms: list[float] = []

        for info in dataset.file_infos:
            offsets = np.asarray(info.get('offsets', []), dtype=np.float64)
            if offsets.size:
                all_offsets.append(offsets)

            n_samples = int(info.get('n_samples', 0))
            dt_sec = float(info.get('dt_sec', 0.0))
            all_tend_ms.append(n_samples * dt_sec * 1000.0)

        if not all_offsets:
            raise RuntimeError('No offsets collected from training dataset.')

        offsets_concat = np.concatenate(all_offsets)
        x95_m = float(np.percentile(offsets_concat, 95))
        t95_ms = float(np.percentile(np.asarray(all_tend_ms, dtype=np.float64), 95))
        return x95_m, t95_ms
    finally:
        dataset.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute robust normalization statistics from the training split.'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Override the dataset root directory (defaults to cfg.data_root).',
    )
    parser.add_argument(
        '--train_field_list',
        type=str,
        default=None,
        help='Override the training field list file (defaults to cfg.train_field_list).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with initialize(config_path='../configs', version_base='1.3'):
        cfg = compose(config_name='base')

    data_root = args.data_root or cfg.data_root
    train_field_list = args.train_field_list or cfg.train_field_list

    x95_m, t95_ms = compute_norm_stats(
        cfg,
        data_root=data_root,
        train_field_list=train_field_list,
    )

    print('norm:')
    print(f'  x95_m: {x95_m:.4f}')
    print(f'  t95_ms: {t95_ms:.4f}')


if __name__ == '__main__':
    main()


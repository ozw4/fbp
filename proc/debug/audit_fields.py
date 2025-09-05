"""Debug script to audit offset fields using audit utilities."""

import argparse
from pathlib import Path

from hydra import compose, initialize
from torch.utils.data import DataLoader, SequentialSampler

from proc.util.audit import audit_offsets_and_mask_coverage
from proc.util.collate import segy_collate
from proc.util.dataset import MaskedSegyGather
from proc.util.rng_util import worker_init_fn


def collect_field_files(list_name: str, data_root: str) -> tuple[list[Path], list[Path]]:
    """Return matching SEG-Y and FB files for each field listed."""
    list_path = Path(__file__).resolve().parent.parent / "configs" / list_name
    with open(list_path) as f:
        fields = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    segy_files: list[Path] = []
    fb_files: list[Path] = []
    for field in fields:
        d = Path(data_root) / field
        segy = sorted(list(d.glob('*.sgy')) + list(d.glob('*.segy')))
        fb = sorted(d.glob('*.npy'))
        if not segy or not fb:
            continue
        segy_files.append(segy[0])
        fb_files.append(fb[0])
    return segy_files, fb_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='base')
    parser.add_argument('--list', default='train_field_list')
    parser.add_argument('--audit-batches', type=int, default=1)
    parser.add_argument('--cov-th', type=float, default=0.5)
    args = parser.parse_args()

    with initialize(config_path='../configs', version_base='1.3'):
        cfg = compose(config_name=args.cfg)

    data_root = Path(cfg.data_root)
    if not data_root.exists():
        print(f"[WARN] data_root {data_root} not found; skipping audit")
        return

    field_list = getattr(cfg, args.list)
    segy_files, fb_files = collect_field_files(field_list, str(data_root))
    if not segy_files:
        print('[WARN] no data files found; skipping audit')
        return

    dataset = MaskedSegyGather(
        segy_files,
        fb_files,
        mask_ratio=cfg.dataset.mask_ratio,
        mask_mode=cfg.dataset.mask_mode,
        mask_noise_std=cfg.dataset.mask_noise_std,
        target_mode=cfg.dataset.target_mode,
        label_sigma=cfg.dataset.label_sigma,
        flip=False,
        augment_time_prob=0.0,
        augment_space_prob=0.0,
        augment_freq_prob=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=SequentialSampler(dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=segy_collate,
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )

    audit_offsets_and_mask_coverage(
        loader,
        cfg.loss.fb_seg,
        max_batches=args.audit_batches,
        cov_threshold=args.cov_th,
    )


if __name__ == '__main__':
    main()

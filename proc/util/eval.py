import logging
from collections.abc import Sequence

import numpy as np
import segyio
import torch
import torch.nn.functional as F

from proc.util.features import make_offset_channel
from proc.util.metrics import prepare_fb_windows, snr_improvement_from_cached_windows
from proc.util.vis import visualize_recon_triplet

from .predict import cover_all_traces_predict

__all__ = ['eval_synthe', 'val_one_epoch_snr']

logger = logging.getLogger(__name__)


def _source_group_scalar_to_scale(scalar: np.ndarray | Sequence[int] | int | None) -> np.ndarray | float:
        """Convert SEG-Y SourceGroupScalar value(s) to numeric scale(s)."""
        if scalar is None:
                return 1.0
        arr = np.asarray(scalar)
        if arr.size == 0:
                return 1.0
        arr = arr.astype(np.int64, copy=False)
        scale = np.ones(arr.shape, dtype=np.float64)
        pos = arr > 0
        neg = arr < 0
        if pos.any():
                scale[pos] = arr[pos].astype(np.float64, copy=False)
        if neg.any():
                scale[neg] = 1.0 / np.abs(arr[neg]).astype(np.float64, copy=False)
        if np.isscalar(scalar) or scale.ndim == 0:
                return float(scale.reshape(-1)[0])
        return scale


def _offsets_from_trace_headers(segy_path: str, trace_indices: Sequence[int]) -> torch.Tensor:
        """Read offsets for ``trace_indices`` from SEG-Y headers."""
        idx = np.asarray(trace_indices, dtype=np.int64)
        if idx.ndim != 1:
                idx = idx.reshape(-1)
        with segyio.open(segy_path, "r", ignore_geometry=True) as f:
                src_x = np.asarray(
                        f.attributes(segyio.TraceField.SourceX)[idx], dtype=np.float64, copy=False
                )
                grp_x = np.asarray(
                        f.attributes(segyio.TraceField.GroupX)[idx], dtype=np.float64, copy=False
                )
                try:
                        scalars_attr = f.attributes(segyio.TraceField.SourceGroupScalar)
                        scalars = scalars_attr[idx]
                        scale = _source_group_scalar_to_scale(scalars)
                except Exception:
                        scale = 1.0
        scale = np.asarray(scale, dtype=np.float64)
        offsets = np.abs(src_x - grp_x) * scale
        return torch.from_numpy(offsets.astype(np.float32, copy=False))


def _normalize_trace_indices(indices, B: int) -> list[list[int]]:
        """Convert metadata trace indices into a list of Python lists."""
        if isinstance(indices, torch.Tensor):
                arr = indices.detach().cpu()
                if arr.ndim == 2 and arr.size(0) == B:
                        return [arr[b].view(-1).to(torch.long).tolist() for b in range(B)]
                if arr.ndim == 1 and B == 1:
                        return [arr.view(-1).to(torch.long).tolist()]
        if isinstance(indices, np.ndarray):
                arr = indices
                if arr.ndim == 2 and arr.shape[0] == B:
                        return [arr[b].astype(np.int64).reshape(-1).tolist() for b in range(B)]
                if arr.ndim == 1 and B == 1:
                        return [arr.astype(np.int64).reshape(-1).tolist()]
        if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
                if len(indices) == B and all(
                        isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)) for elem in indices
                ):
                        out: list[list[int]] = []
                        for elem in indices:
                                if isinstance(elem, torch.Tensor):
                                        out.append(elem.detach().cpu().view(-1).to(torch.long).tolist())
                                elif isinstance(elem, np.ndarray):
                                        out.append(elem.astype(np.int64).reshape(-1).tolist())
                                else:
                                        out.append([int(x) for x in elem])
                        return out
                if B == 1:
                        return [[int(x) for x in indices]]
        raise TypeError('Unsupported indices type for offset synthesis')


def _prepare_batch_offsets(meta: dict, B: int, H: int) -> torch.Tensor:
        """Ensure offsets tensor exists, synthesizing from headers when needed."""
        raw_offsets = meta.get('offsets')
        offsets = torch.zeros((B, H), dtype=torch.float32)
        if raw_offsets is not None:
                if isinstance(raw_offsets, torch.Tensor):
                        raw_tensor = raw_offsets.detach().to(torch.float32)
                else:
                        raw_tensor = torch.as_tensor(raw_offsets, dtype=torch.float32)
                if raw_tensor.ndim == 3 and raw_tensor.size(1) == 1:
                        raw_tensor = raw_tensor.squeeze(1)
                if raw_tensor.shape == (B, H):
                        offsets.copy_(raw_tensor)
                elif raw_tensor.ndim == 1 and raw_tensor.numel() == H and B == 1:
                        offsets[0] = raw_tensor
                elif raw_tensor.numel() == B * H:
                        offsets.copy_(raw_tensor.view(B, H))
                else:
                        b_lim = min(B, raw_tensor.shape[0])
                        h_lim = min(H, raw_tensor.shape[-1])
                        view = raw_tensor.reshape(-1, raw_tensor.shape[-1])
                        offsets[:b_lim, :h_lim] = view[:b_lim, :h_lim]
        need_synth = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
                row = offsets[b]
                if not torch.isfinite(row).all() or torch.all(row == 0):
                        need_synth[b] = True
        if need_synth.any():
                file_paths = None
                for key in ('file_path', 'file_paths', 'paths', 'path'):
                        if key in meta:
                                file_paths = meta[key]
                                break
                if file_paths is None:
                        raise KeyError('file_path metadata is required to synthesize offsets')
                if isinstance(file_paths, (str, bytes)):
                        file_paths = [file_paths] * B
                indices_meta = None
                for key in ('indices', 'trace_indices', 'trace_idxs', 'idx', 'trace_idx'):
                        if key in meta:
                                indices_meta = meta[key]
                                break
                if indices_meta is None:
                        raise KeyError('trace indices metadata is required to synthesize offsets')
                indices_list = _normalize_trace_indices(indices_meta, B)
                synth_count = 0
                for b in range(B):
                        if not need_synth[b]:
                                continue
                        synth_count += 1
                        offsets_b = _offsets_from_trace_headers(file_paths[b], indices_list[b])
                        if offsets_b.numel() != H:
                                raise ValueError(
                                        f'Offsets length mismatch for {file_paths[b]}: '
                                        f'{offsets_b.numel()} vs {H}'
                                )
                        offsets[b] = offsets_b
                if synth_count > 0:
                        if logger.isEnabledFor(logging.DEBUG):
                                sample_b = int(torch.nonzero(need_synth, as_tuple=False)[0].item())
                                preview = offsets[sample_b, : min(5, H)].tolist()
                                logger.debug(
                                        'Synthesized offsets from SEG-Y headers for %d/%d shots (example %s: %s)',
                                        synth_count,
                                        B,
                                        file_paths[sample_b],
                                        preview,
                                )
                        else:
                                logger.debug(
                                        'Synthesized offsets from SEG-Y headers for %d/%d shots', synth_count, B
                                )
        if torch.any(offsets < 0):
                raise ValueError('Offsets derived from headers must be non-negative')
        meta['offsets'] = offsets
        return offsets


def _should_use_offset_input(cfg_snr) -> bool:
        if getattr(cfg_snr, 'use_offset_input', False):
                return True
        if getattr(cfg_snr, 'use_offsets', False):
                return True
        cfg_model = getattr(cfg_snr, 'model', None)
        if cfg_model is not None and getattr(cfg_model, 'use_offset_input', False):
                return True
        return False


class _ModelWithOffsetChannel:
        """Callable wrapper to append offset channel before model inference."""

        def __init__(self, base_model: torch.nn.Module, offsets: torch.Tensor):
                self._model = base_model
                self._offsets = offsets.detach().to(torch.float32)

        def __call__(self, x_amp: torch.Tensor) -> torch.Tensor:
                Bp = x_amp.shape[0]
                offs = self._offsets.expand(Bp, -1)
                offs_ch = make_offset_channel(x_amp, offs)
                x_in = torch.cat([x_amp, offs_ch], dim=1)
                return self._model(x_in)


def val_one_epoch_snr(
        model,
        val_loader,
        device,
        cfg_snr,
        visualize: bool = False,
        viz_batches: tuple[int, ...] = (0,),
        out_dir=None,
        writer=None,
        epoch: int | None = None,
        is_main_process: bool = True,
):
        """Evaluate SNR improvement over validation loader."""
        import matplotlib.pyplot as plt

        model.eval()
        use_offset_input = _should_use_offset_input(cfg_snr)
        all_in, all_out, all_imp, all_vf = [], [], [], []
        for i, (x_masked, x_orig, _, meta) in enumerate(val_loader):
                del x_masked  # not needed for evaluation
                x_orig = x_orig.to(device, non_blocking=True)
                fb_idx = meta['fb_idx'].to(device)
                B, _, H, W = x_orig.shape
                if use_offset_input:
                        offsets = _prepare_batch_offsets(meta, B, H)
                        seed_base = getattr(cfg_snr, 'seed', None)
                        y_parts = []
                        for b in range(B):
                                wrapper = _ModelWithOffsetChannel(model, offsets[b : b + 1])
                                sample_seed = None if seed_base is None else int(seed_base) + b
                                y_b = cover_all_traces_predict(
                                        wrapper,
                                        x_orig[b : b + 1],
                                        mask_ratio=cfg_snr.mask_ratio_for_eval,
                                        noise_std=getattr(cfg_snr, 'noise_std', 1.0),
                                        mask_noise_mode=getattr(cfg_snr, 'mask_noise_mode', 'replace'),
                                        use_amp=True,
                                        device=device,
                                        seed=sample_seed,
                                        passes_batch=cfg_snr.passes_batch,
                                )
                                y_parts.append(y_b)
                        y_full = torch.cat(y_parts, dim=0) if y_parts else torch.empty_like(x_orig)
                else:
                        y_full = cover_all_traces_predict(
                                model,
                                x_orig,
                                mask_ratio=cfg_snr.mask_ratio_for_eval,
                                noise_std=getattr(cfg_snr, 'noise_std', 1.0),
                                mask_noise_mode=getattr(cfg_snr, 'mask_noise_mode', 'replace'),
                                use_amp=True,
                                device=device,
                                seed=getattr(cfg_snr, 'seed', None),
                                passes_batch=cfg_snr.passes_batch,
                        )
                cache = prepare_fb_windows(
                        fb_idx,
                        W=x_orig.shape[-1],
                        pre_len=cfg_snr.pre_len,
                        post_len=cfg_snr.post_len,
                        guard=cfg_snr.guard,
                )
                res = snr_improvement_from_cached_windows(
                        x_orig, y_full, cache, reduction='median'
                )
                all_in.append(res['snr_in_db'].item())
                all_out.append(res['snr_out_db'].item())
                all_imp.append(res['snr_improve_db'].item())
                all_vf.append(res['valid_frac'].item())
                if visualize and is_main_process and (i in viz_batches):
                        gs = int(epoch) if isinstance(epoch, int) else 0
                        fig = visualize_recon_triplet(
                                x_orig,
                                y_full,
                                fb_idx=meta['fb_idx'],
                                b=0,
                                transpose=True,
                                prefix=f'batch{i:04d}',
                                writer=writer,
                                global_step=gs,
                        )
                        plt.close(fig)
        return {
                'snr_in_db': float(np.median(all_in)),
                'snr_out_db': float(np.median(all_out)),
                'snr_improve_db': float(np.median(all_imp)),
                'valid_frac': float(np.mean(all_vf)),
        }


def eval_synthe(x_clean, pred, device=None):
        """Compute MSE, MAE and PSNR for synthetic data."""
        mses, maes, psnrs = [], [], []
        for p, gt in zip(pred, x_clean, strict=False):
                if device is not None:
                        p, gt = p.to(device), gt.to(device)
                mse = F.mse_loss(p, gt).item()
                mae = F.l1_loss(p, gt).item()
                psnr = -10.0 * torch.log10(F.mse_loss(p, gt)).item()
                mses.append(mse)
                maes.append(mae)
                psnrs.append(psnr)
        return {
                'mse': float(sum(mses) / len(mses)),
                'mae': float(sum(maes) / len(maes)),
                'psnr': float(sum(psnrs) / len(psnrs)),
                'num_shots': len(x_clean),
        }

import torch
import torch.nn.functional as F
from util.loss import shift_robust_l2_pertrace_vec

__all__ = ["compute_loss", "make_criterion", "shift_robust_l2_pertrace_vec"]

def compute_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None, cfg_loss=None):
    """Compute training loss.

    Uses standard MSE by default, and switches to shift-robust MSE when
    ``cfg_loss.shift_robust`` is True. The existing implementation
    ``shift_robust_l2_pertrace_vec`` is reused for the robust variant.
    """
    shift_robust = bool(getattr(cfg_loss, "shift_robust", False))
    max_shift = int(getattr(cfg_loss, "max_shift", 5))
    if shift_robust:
        return shift_robust_l2_pertrace_vec(
            pred, target, mask=mask, max_shift=max_shift, reduction="mean"
        )
    # standard masked/unmasked MSE
    if mask is not None:
        if mask.dim() != 4:
            mask = mask.view(pred.size(0), 1, pred.size(2), pred.size(3))
        if mask.size(1) == 1:
            mask = mask.expand(-1, pred.size(1), -1, -1)
        diff2 = (pred - target) ** 2
        masked = diff2 * mask
        return masked.sum() / mask.sum().clamp_min(1e-8)
    return F.mse_loss(pred, target)

def make_criterion(cfg_loss):
    """Return a criterion compatible with ``train_one_epoch``."""
    def _criterion(pred, target, *, mask=None, max_shift=None, reduction="mean"):
        return compute_loss(pred, target, mask=mask, cfg_loss=cfg_loss)
    return _criterion

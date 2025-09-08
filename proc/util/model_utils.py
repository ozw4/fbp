from torch import nn

__all__ = ["inflate_input_convs_to_2ch"]

def _inflate_conv(conv: nn.Conv2d, *, verbose: bool, init_mode: str) -> None:
    """Inflate a Conv2d with 1 input channel to 2 channels.

    Parameters
    ----------
    conv : nn.Conv2d
        Target convolution layer.
    verbose : bool
        Whether to print a message when inflating.
    init_mode : str
        Weight init mode. Currently only 'duplicate' is supported.

    """
    if conv.in_channels != 1:
        return
    w_old = conv.weight.data  # (out_c,1,kH,kW)
    if init_mode == "duplicate":
        w_new = w_old.repeat(1, 2, 1, 1).clone()
    else:
        w_new = w_old.repeat(1, 2, 1, 1).clone()
    conv.weight = nn.Parameter(w_new)
    conv.in_channels = 2
    if verbose:
        print(f"[inflate] set {conv.__class__.__name__} in_channels=2")

def inflate_input_convs_to_2ch(model: nn.Module, *, verbose: bool = True, init_mode: str = "duplicate") -> None:
    """Inflate the first conv(s) that receive the raw input to accept 2 channels.

    - Targets:
      (a) The very first conv in model.pre_down[0].
      (b) The backbone's first input conv (stem_0 or patch_embed.proj) if in_channels == 1.

    Parameters
    ----------
    model : nn.Module
        Model whose input convs will be inflated.
    verbose : bool, default True
        Whether to print messages.
    init_mode : str, default "duplicate"
        Weight initialization mode for the new channel.

    """
    # (a) first conv in pre_down[0]
    if hasattr(model, "pre_down") and len(model.pre_down) > 0:
        first_block = model.pre_down[0]
        for m in first_block.modules():
            if isinstance(m, nn.Conv2d):
                _inflate_conv(m, verbose=verbose, init_mode=init_mode)
                break

    # (b) backbone's first conv
    backbone = getattr(model, "backbone", None)
    conv = None
    if backbone is not None:
        if hasattr(backbone, "stem_0") and isinstance(backbone.stem_0, nn.Conv2d):
            conv = backbone.stem_0
        elif (
            hasattr(backbone, "patch_embed")
            and hasattr(backbone.patch_embed, "proj")
            and isinstance(backbone.patch_embed.proj, nn.Conv2d)
        ):
            conv = backbone.patch_embed.proj
    if conv is not None:
        _inflate_conv(conv, verbose=verbose, init_mode=init_mode)

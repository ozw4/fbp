# proc/util/model_utils.py
from __future__ import annotations

import torch
from torch import nn


def _dup_or_init_like(
	old_w: torch.Tensor, out_ch: int, in_ch: int, mode: str
) -> torch.Tensor:
	"""Create new (out_ch, in_ch, kH, kW) from old_w (o, i, kH, kW)."""
	kH, kW = old_w.shape[-2], old_w.shape[-1]
	device, dtype = old_w.device, old_w.dtype
	new_w = torch.zeros((out_ch, in_ch, kH, kW), device=device, dtype=dtype)

	if mode == 'zeros':
		return new_w

	if mode == 'random':
		# Kaiming-like
		nn.init.kaiming_normal_(new_w, nonlinearity='relu')
		return new_w

	# "duplicate" or fallback: tile the single-channel kernels
	# - if old has shape (O, 1, kH, kW) or (1, 1, kH, kW), we broadcast to (out_ch, in_ch, kH, kW)
	base = old_w
	if base.shape[0] == 1:
		base = base.expand(out_ch, base.shape[1], kH, kW)
	if base.shape[1] == 1:
		base = base.expand(base.shape[0], in_ch, kH, kW)

	# if old already has >1 in/out, just center-crop/avg to fit
	base = base[:out_ch, :in_ch].clone()
	return base


def _inflate_conv_in_to_2(conv: nn.Conv2d, *, verbose: bool, init_mode: str) -> None:
	"""Make conv.in_channels = 2 (keep out_channels)."""
	if conv.in_channels == 2:
		if verbose:
			print(f'[inflate] keep in=2 for {conv}')
		return
	assert conv.in_channels == 1, f'in_channels must be 1 or 2, got {conv.in_channels}'
	out_ch = conv.out_channels
	new_conv = nn.Conv2d(
		in_channels=2,
		out_channels=out_ch,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		dilation=conv.dilation,
		groups=conv.groups,
		bias=(conv.bias is not None),
		padding_mode=conv.padding_mode,
		device=conv.weight.device,
		dtype=conv.weight.dtype,
	)
	with torch.no_grad():
		new_conv.weight.copy_(_dup_or_init_like(conv.weight.data, out_ch, 2, init_mode))
		if conv.bias is not None:
			new_conv.bias.copy_(conv.bias.data)
	# in-place swap
	conv.weight = new_conv.weight
	conv.bias = new_conv.bias
	conv.in_channels = new_conv.in_channels
	if verbose:
		print(f'[inflate] {type(conv).__name__}: in 1→2')


def _replace_seq_conv_bn_to_2x(
	seq: nn.Sequential,
	conv_idx: int,
	bn_idx: int | None,
	*,
	verbose: bool,
	init_mode: str,
) -> None:
	"""Replace seq[conv_idx] (Conv2d) to in=2,out=2 and BN to num_features=2."""
	old_conv: nn.Conv2d = seq[conv_idx]
	assert isinstance(old_conv, nn.Conv2d)
	new_conv = nn.Conv2d(
		in_channels=2,
		out_channels=2,
		kernel_size=old_conv.kernel_size,
		stride=old_conv.stride,
		padding=old_conv.padding,
		dilation=old_conv.dilation,
		groups=old_conv.groups,
		bias=(old_conv.bias is not None),
		padding_mode=old_conv.padding_mode,
		device=old_conv.weight.device,
		dtype=old_conv.weight.dtype,
	)
	with torch.no_grad():
		new_conv.weight.copy_(_dup_or_init_like(old_conv.weight.data, 2, 2, init_mode))
		if old_conv.bias is not None:
			# duplicate bias if needed
			b = old_conv.bias.data
			if b.numel() == 1:
				new_conv.bias.copy_(b.expand(2))
			else:
				new_conv.bias.copy_(b[:2])

	seq[conv_idx] = new_conv
	if verbose:
		print(f'[inflate] {seq.__class__.__name__}[{conv_idx}]: (in,out) → (2,2)')

	if (
		bn_idx is not None
		and 0 <= bn_idx < len(seq)
		and isinstance(seq[bn_idx], (nn.BatchNorm2d, nn.SyncBatchNorm))
	):
		old_bn = seq[bn_idx]
		new_bn = type(old_bn)(
			2,
			eps=old_bn.eps,
			momentum=old_bn.momentum,
			affine=old_bn.affine,
			track_running_stats=old_bn.track_running_stats,
			device=old_bn.weight.device,
			dtype=old_bn.weight.dtype,
		)
		with torch.no_grad():
			if old_bn.affine:
				w = old_bn.weight.data
				b = old_bn.bias.data
				if w.numel() == 1:
					new_bn.weight.copy_(w.expand(2))
					new_bn.bias.copy_(b.expand(2))
				else:
					new_bn.weight.copy_(w[:2])
					new_bn.bias.copy_(b[:2])
			if old_bn.track_running_stats:
				rm = old_bn.running_mean
				rv = old_bn.running_var
				if rm.numel() == 1:
					new_bn.running_mean.copy_(rm.expand(2))
					new_bn.running_var.copy_(rv.expand(2))
				else:
					new_bn.running_mean.copy_(rm[:2])
					new_bn.running_var.copy_(rv[:2])
		seq[bn_idx] = new_bn
		if verbose:
			print(f'[inflate] {seq.__class__.__name__}[{bn_idx}]: BN channels → 2')


def _find_backbone_first_conv(bb: nn.Module) -> Tuple[nn.Conv2d | None, str]:
	"""Find the first Conv2d that consumes the raw feature map inside a backbone.
	Supports common patterns:
	  - stem_0 (ConvNeXt/timm FeatureListNet variants)
	  - stem.conv / stem (Caformer系 Stem)
	  - patch_embed.proj (ViT/Hybrid)
	  - conv1 (ResNet系)
	As a fallback, return the first nn.Conv2d found by named_modules() preorder.
	Returns (conv, qualified_name) or (None, '').
	"""
	# 1) Named attribute candidates by priority
	try_candidates = []
	if hasattr(bb, 'stem_0'):
		try_candidates.append(('stem_0', bb.stem_0))
	if hasattr(bb, 'stem'):
		stem = bb.stem
		try_candidates.append(('stem', stem))
		if hasattr(stem, 'conv'):
			try_candidates.append(('stem.conv', stem.conv))
		if hasattr(stem, 'proj'):
			try_candidates.append(('stem.proj', stem.proj))
	if hasattr(bb, 'patch_embed') and hasattr(bb.patch_embed, 'proj'):
		try_candidates.append(('patch_embed.proj', bb.patch_embed.proj))
	if hasattr(bb, 'conv1'):
		try_candidates.append(('conv1', bb.conv1))

	for qname, obj in try_candidates:
		if isinstance(obj, nn.Conv2d):
			return obj, f'backbone.{qname}'

	# 2) Generic fallback: first Conv2d by preorder (skip grouped/depthwise)
	for qname, m in bb.named_modules():
		if isinstance(m, nn.Conv2d) and m.groups == 1:
			return m, f'backbone.{qname}'

	return None, ''


def _inflate_conv_in_channels(
	conv: nn.Conv2d,
	target_in_ch: int,
	*,
	verbose: bool,
	init_mode: str,
	name: str,
) -> bool:
	"""Inflate a Conv2d's input channels to `target_in_ch` in-place, preserving out_channels.
	- init_mode: 'duplicate' | 'random' | 'zeros'
	  * 'duplicate': tile existing kernels across new channels, then scale by (old_in / new_in)
	  * 'random'   : kaiming_normal_ for the whole new weight tensor
	  * 'zeros'    : all zeros
	Returns:
	  True if weights were changed, False if already matched (or skipped).
	"""
	assert isinstance(conv, nn.Conv2d)
	old_in = conv.in_channels
	out_ch, kH, kW = conv.out_channels, conv.kernel_size[0], conv.kernel_size[1]

	if old_in == target_in_ch:
		if verbose:
			print(f'[inflate] keep in={old_in} for {name}')
		return False

	if old_in > target_in_ch:
		if verbose:
			print(f'[inflate][WARN] skip {name}: in={old_in} > target={target_in_ch}')
		return False

	# depthwise/grouped conv は対象外
	assert conv.groups == 1, (
		f'Cannot inflate grouped/depthwise convs (groups={conv.groups})'
	)

	# 重み生成ロジック
	if '_make_inflated_weight' in globals():
		new_weight = _make_inflated_weight(conv, target_in_ch, init_mode)
	else:
		# フォールバック: 依存関数が無い環境でも動くようにここで生成
		device, dtype = conv.weight.device, conv.weight.dtype
		if init_mode == 'zeros':
			new_weight = torch.zeros(
				(out_ch, target_in_ch, kH, kW), device=device, dtype=dtype
			)
		elif init_mode == 'random':
			new_weight = torch.empty(
				(out_ch, target_in_ch, kH, kW), device=device, dtype=dtype
			)
			nn.init.kaiming_normal_(new_weight, nonlinearity='relu')
		elif init_mode == 'duplicate':
			old_w = conv.weight.data  # (out_ch, old_in, kH, kW)
			repeats = target_in_ch // old_in
			remainder = target_in_ch % old_in
			chunks = []
			if repeats > 0:
				chunks.append(old_w.repeat(1, repeats, 1, 1))
			if remainder > 0:
				mean_w = old_w.mean(dim=1, keepdim=True)
				chunks.append(mean_w.repeat(1, remainder, 1, 1))
			new_weight = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
			# fan-in を揃えるためスケール
			new_weight = new_weight * (float(old_in) / float(target_in_ch))
		else:
			raise ValueError(f"Unsupported init_mode '{init_mode}'")

	# in-place swap
	conv.in_channels = target_in_ch
	conv.weight = nn.Parameter(new_weight)

	if verbose:
		print(f'[inflate] {name}: in {old_in}→{target_in_ch}')

	return True


def inflate_input_convs_to_2ch(
	model: nn.Module,
	*,
	verbose: bool = True,
	init_mode: str = 'duplicate',
	fix_predown: bool = True,
	fix_backbone: bool = True,
) -> None:
	"""Make the *raw-input path* truly 2ch end-to-end:
	  - pre_down[0] and pre_down[1] convs → (in=2, out=2) ＋ 対応BNを2ch化
	  - backbone 最初の Conv を汎用に特定（stem_0 / stem.conv / patch_embed.proj / conv1 など）し in を 2ch にinflate

	Tips:
	  - 既存 1ch 重みを複製して2chへ展開（init_mode='duplicate' 推奨）
	  - Caformer 等でも最初の Conv を自動検出
	"""
	# (A) pre_down の 2段を 2→2 に強制
	if fix_predown and hasattr(model, 'pre_down') and len(model.pre_down) > 0:
		for blk_idx in range(min(2, len(model.pre_down))):
			seq = model.pre_down[blk_idx]
			# 期待構造: Sequential(Conv, BN, ReLU)
			# Conv を (2,2) に、BN を 2 にそろえる
			if isinstance(seq, nn.Sequential):
				# conv は大抵 index 0, BN は index 1
				_replace_seq_conv_bn_to_2x(
					seq, conv_idx=0, bn_idx=1, verbose=verbose, init_mode=init_mode
				)

	# (B) backbone 最初の Conv の in を 2ch に
	if fix_backbone and hasattr(model, 'backbone'):
		bb = model.backbone
		conv = None
		conv_name = ''

		# 汎用: _find_backbone_first_conv があれば使う
		if '_find_backbone_first_conv' in globals():
			conv, conv_name = _find_backbone_first_conv(bb)

		# 互換: 見つからなければ従来の分岐
		if conv is None:
			if hasattr(bb, 'stem_0') and isinstance(bb.stem_0, nn.Conv2d):
				conv, conv_name = bb.stem_0, 'backbone.stem_0'
			elif (
				hasattr(bb, 'patch_embed')
				and hasattr(bb.patch_embed, 'proj')
				and isinstance(bb.patch_embed.proj, nn.Conv2d)
			):
				conv, conv_name = bb.patch_embed.proj, 'backbone.patch_embed.proj'

		if conv is not None:
			if conv.in_channels != 2:
				# 1→2 だけでなく任意 in→2 を安全に対応
				_inflate_conv_in_channels(
					conv,
					2,
					verbose=verbose,
					init_mode=init_mode,
					name=conv_name or type(conv).__name__,
				)
			elif verbose:
				print(f'[inflate] keep in=2 for {conv_name or type(conv).__name__}')
		elif verbose:
			print('[inflate][WARN] backbone first Conv2d not found')

	if verbose:
		print('[inflate] done.')


def _register_grad_mask_for_old_in_channels(conv: nn.Conv2d, old_in_ch: int):
	"""conv.weight: (out_ch, in_ch, kH, kW)
	旧 in-ch 部分 [:, :old_in_ch, ...] の勾配を 0 にして凍結する。
	"""
	assert isinstance(conv, nn.Conv2d)
	assert conv.weight.size(1) >= old_in_ch

	# マスクを buffer に持たせる（後からall-onesに差し替えれば解除できる）
	mask = torch.zeros_like(conv.weight)
	mask[:, old_in_ch:, :, :] = 1.0  # 新チャンネルのみ学習
	conv.register_buffer('_grad_mask_in_old', mask, persistent=False)

	# 既存のフックがあれば外す
	if hasattr(conv, '_grad_mask_handle') and conv._grad_mask_handle is not None:
		try:
			conv._grad_mask_handle.remove()
		except Exception:
			pass
		conv._grad_mask_handle = None

	def _hook(grad):
		g = grad
		m = getattr(conv, '_grad_mask_in_old', None)
		if m is not None:
			g = g * m
		return g

	conv._grad_mask_handle = conv.weight.register_hook(_hook)


def _remove_grad_mask(conv: nn.Conv2d):
	if hasattr(conv, '_grad_mask_handle') and conv._grad_mask_handle is not None:
		try:
			conv._grad_mask_handle.remove()
		except Exception:
			pass
		conv._grad_mask_handle = None
	if hasattr(conv, '_grad_mask_in_old'):
		conv._grad_mask_in_old = None


def find_input_convs_for_inflation(model: nn.Module):
	"""あなたのNetAEに合わせて、入力を最初に受けるConvを列挙。
	- pre_down[0] の先頭Conv
	- pre_down[1] の先頭Conv（あれば）
	- backbone.stem_0 も対象
	"""
	convs = []
	if hasattr(model, 'pre_down') and len(model.pre_down) > 0:
		for idx in (0, 1):  # あれば 0,1 を見る
			if idx < len(model.pre_down):
				for m in model.pre_down[idx].modules():
					if isinstance(m, nn.Conv2d):
						convs.append(m)
						break
	if hasattr(model, 'backbone') and hasattr(model.backbone, 'stem_0'):
		if isinstance(model.backbone.stem_0, nn.Conv2d):
			convs.append(model.backbone.stem_0)
	# 重複除去
	seen = set()
	uniq = []
	for c in convs:
		if id(c) not in seen:
			uniq.append(c)
			seen.add(id(c))
	return uniq


def freeze_original_in_channels(model: nn.Module, old_in_ch: int = 1):
	"""旧 in-ch を凍結（新規追加チャンネルのみ学習）するフックを登録。"""
	for conv in find_input_convs_for_inflation(model):
		if conv.in_channels >= old_in_ch + 1:
			_register_grad_mask_for_old_in_channels(conv, old_in_ch)


def unfreeze_all_inflated_convs(model: nn.Module):
	"""凍結解除：勾配マスクを外す。"""
	for conv in find_input_convs_for_inflation(model):
		_remove_grad_mask(conv)

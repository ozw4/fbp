# %%
import timm
import torch
import torch.nn.functional as F
from torch import nn


def adjust_first_conv_padding(backbone: nn.Module, padding=(1, 1)):
	"""backboneの最初のConvだけpaddingを上書き"""
	for m in backbone.modules():
		if isinstance(m, nn.Conv2d):
			m.padding = padding

			print(f'Adjusted first Conv2d padding to {padding}')
			break


def adapt_stage_strides(backbone: nn.Module, stage_strides: list[tuple[int, int]]):
	"""Apply anisotropic strides per stage to the backbone.

	Parameters
	----------
	backbone: nn.Module
		Encoder backbone whose strided convolutions will be modified.
	stage_strides: list[tuple[int, int]]
		List of strides ``(sH, sW)`` applied in order of appearance to
		convolutions with ``stride>1``.

	Raises
	------
	ValueError
		If ``stage_strides`` has more elements than the number of
		detected strided convolutions.

	"""
	convs: list[nn.Conv2d] = [
		m
		for m in backbone.modules()
		if isinstance(m, nn.Conv2d) and (m.stride[0] > 1 or m.stride[1] > 1)
	]

	if len(stage_strides) > len(convs):
		raise ValueError('stage_strides longer than detected stride layers')

	for conv, s in zip(convs, stage_strides, strict=False):
		conv.stride = s


#################
# Decoder 周りは既存
#################


class ConvBnAct2d(nn.Module):
	def __init__(
		self,
		in_channels,
		out_channels,
		kernel_size,
		padding: int = 0,
		stride: int = 1,
		norm_layer: nn.Module = nn.Identity,
		act_layer: nn.Module = nn.ReLU,
	):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size,
			stride=stride,
			padding=padding,
			bias=False,
		)
		self.norm = (
			norm_layer(out_channels) if norm_layer != nn.Identity else nn.Identity()
		)
		self.act = act_layer(inplace=True)

	def forward(self, x):
		return self.act(self.norm(self.conv(x)))


class SCSEModule2d(nn.Module):
	def __init__(self, in_channels, reduction=16):
		super().__init__()
		self.cSE = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, in_channels // reduction, 1),
			nn.Tanh(),
			nn.Conv2d(in_channels // reduction, in_channels, 1),
			nn.Sigmoid(),
		)
		self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

	def forward(self, x):
		return x * self.cSE(x) + x * self.sSE(x)


class Attention2d(nn.Module):
	def __init__(self, name, **params):
		super().__init__()
		if name is None:
			self.attention = nn.Identity(**params)
		elif name == 'scse':
			self.attention = SCSEModule2d(**params)
		else:
			raise ValueError(f'Attention {name} is not implemented')

	def forward(self, x):
		return self.attention(x)


class DecoderBlock2d(nn.Module):
	"""Decoder block with interpolation-based upsampling."""

	def __init__(
		self,
		in_channels,
		skip_channels,
		out_channels,
		norm_layer: nn.Module = nn.Identity,
		attention_type: str = None,
		intermediate_conv: bool = False,
		upsample_mode: str = 'bilinear',
		scale_factor: int | tuple[int, int] = 2,
	):
		super().__init__()
		self.upsample_mode = upsample_mode
		self.scale_factor = scale_factor

		if intermediate_conv:
			k = 3
			c = skip_channels if skip_channels != 0 else in_channels
			self.intermediate_conv = nn.Sequential(
				ConvBnAct2d(c, c, k, k // 2),
				ConvBnAct2d(c, c, k, k // 2),
			)
		else:
			self.intermediate_conv = None

		self.attention1 = Attention2d(
			name=attention_type, in_channels=in_channels + skip_channels
		)
		self.conv1 = ConvBnAct2d(
			in_channels + skip_channels, out_channels, 3, 1, norm_layer=norm_layer
		)
		self.conv2 = ConvBnAct2d(
			out_channels, out_channels, 3, 1, norm_layer=norm_layer
		)
		self.attention2 = Attention2d(name=attention_type, in_channels=out_channels)

	def forward(self, x, skip=None):
		if skip is not None:
			x = self._interpolate(x, size=skip.shape[-2:])
		else:
			x = self._interpolate(x, scale_factor=self.scale_factor)
		if self.intermediate_conv is not None:
			if skip is not None:
				skip = self.intermediate_conv(skip)
			else:
				x = self.intermediate_conv(x)
		if skip is not None:
			x = self.attention1(torch.cat([x, skip], dim=1))
		x = self.conv2(self.conv1(x))
		return self.attention2(x)

	def _interpolate(self, x, size=None, scale_factor=None):
		kwargs = {}
		if self.upsample_mode in ('bilinear', 'bicubic'):
			kwargs['align_corners'] = False
		return F.interpolate(
			x,
			size=size,
			scale_factor=scale_factor,
			mode=self.upsample_mode,
			**kwargs,
		)


class UnetDecoder2d(nn.Module):
	def __init__(
		self,
		encoder_channels: tuple[int],
		skip_channels: tuple[int] = None,
		decoder_channels: tuple = (256, 128, 64, 32),
		scale_factors: tuple = (2, 2, 2, 2),
		norm_layer: nn.Module = nn.Identity,
		attention_type: str = 'scse',
		intermediate_conv: bool = True,
		upsample_mode: str = 'bilinear',
	):
		super().__init__()

		# 期待段数 = encoder_levels - 1
		need = len(encoder_channels) - 1

		# --- decoder_channels を need に合わせる ---
		dec = list(decoder_channels)
		if len(dec) < need:
			dec += [dec[-1]] * (need - len(dec))  # 末尾を繰り返して延長
		elif len(dec) > need:
			dec = dec[:need]  # 余剰をカット
		decoder_channels = tuple(dec)
		self.decoder_channels = decoder_channels

		# --- scale_factors も need に合わせる（★これが今回の修正ポイント） ---
		sf = list(
			scale_factors
			if isinstance(scale_factors, (list, tuple))
			else [scale_factors]
		)
		if len(sf) < len(decoder_channels):
			sf += [sf[-1]] * (len(decoder_channels) - len(sf))
		elif len(sf) > len(decoder_channels):
			sf = sf[: len(decoder_channels)]
		self.scale_factors = tuple(sf)

		# skip_channels 安全化
		if skip_channels is None:
			skip_channels = list(encoder_channels[1:]) + [0]
		if len(skip_channels) < len(decoder_channels):
			skip_channels += [0] * (len(decoder_channels) - len(skip_channels))
		else:
			skip_channels = skip_channels[: len(decoder_channels)]

		in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])

		self.blocks = nn.ModuleList(
			[
				DecoderBlock2d(
					ic,
					sc,
					dc,
					norm_layer,
					attention_type,
					intermediate_conv,
					upsample_mode,
					self.scale_factors[i],  # ← 調整後を使う
				)
				for i, (ic, sc, dc) in enumerate(
					zip(in_channels, skip_channels, decoder_channels, strict=False)
				)
			]
		)

	def forward(self, feats: list[torch.Tensor]):
		res = [feats[0]]
		feats = feats[1:]
		for i, b in enumerate(self.blocks):
			skip = feats[i] if i < len(feats) else None
			res.append(b(res[-1], skip=skip))
		return res


class SegmentationHead2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
		)

	def forward(self, x):
		return self.conv(x)


#################
# Net（汎用 Stem & MAE向け出力）
#################


class NetAE(nn.Module):
	"""timm バックボーン + U-Net デコーダの汎用ネットワーク。

	timm バックボーンの前に Conv+BN+ReLU の前段ステージを任意段数挿入できる。
	SAME などの動的パディングはモデル外（データローダ側）で行うこと。

	"""

	def __init__(
		self,
		backbone: str,
		in_chans: int = 1,
		out_chans: int = 1,
		pretrained: bool = True,
		stage_strides: list[tuple[int, int]] | None = None,
		extra_stages: int = 0,
		extra_stage_strides: tuple[tuple[int, int], ...] | None = None,
		extra_stage_channels: tuple[int, ...] | None = None,
		extra_stage_use_bn: bool = True,
		pre_stages: int = 0,
		pre_stage_strides: tuple[tuple[int, int], ...] | None = None,
		pre_stage_kernels: tuple[int, ...] | None = None,
		pre_stage_channels: tuple[int, ...] | None = None,
		pre_stage_use_bn: bool = True,
		# decoder オプション
		decoder_channels: tuple = (256, 128, 64, 32),
		decoder_scales: tuple = (2, 2, 2, 2),
		upsample_mode: str = 'bilinear',
		attention_type: str = 'scse',
		intermediate_conv: bool = True,
	):
		super().__init__()

		# 前段のダウンサンプル
		self.pre_down = nn.ModuleList()
		self.pre_out_channels = []  # ★追加：各pre段の出力chリスト
		c_in = in_chans
		kernels = list(pre_stage_kernels or [])
		strides = list(pre_stage_strides or [])
		channels = list(pre_stage_channels or [])
		for i in range(pre_stages):
			k = kernels[i] if i < len(kernels) else 3
			s = strides[i] if i < len(strides) else (1, 1)
			p = k // 2
			c_out = channels[i] if i < len(channels) else c_in
			block = [nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False)]
			if pre_stage_use_bn:
				block.append(nn.BatchNorm2d(c_out))
			block.append(nn.ReLU(inplace=True))
			self.pre_down.append(nn.Sequential(*block))
			self.pre_out_channels.append(c_out)  # ★控える
			c_in = c_out
		pre_out_ch = c_in
		# Encoder (timm features_only)
		self.backbone = timm.create_model(
			backbone,
			in_chans=pre_out_ch,
			pretrained=pretrained,
			features_only=True,
			drop_path_rate=0.0,
		)
		if stage_strides is not None:
			adapt_stage_strides(self.backbone, stage_strides)

		# 追加のダウンサンプル段
		self.extra_down = nn.ModuleList()
		# 1) backbone のチャンネル列（深い→浅い）
		ecs_base = [fi['num_chs'] for fi in self.backbone.feature_info][::-1]

		# 2) extra_down を作る（最深の上に積む）
		self.extra_down = nn.ModuleList()
		c_in = ecs_base[0] if ecs_base else 0
		extra_out_channels: list[int] = []

		extra_strides = list(extra_stage_strides or [])
		if len(extra_strides) < extra_stages:
			extra_strides += [(2, 2)] * (extra_stages - len(extra_strides))
		extra_channels = list(extra_stage_channels or [])

		for i in range(extra_stages):
			stride = extra_strides[i]
			c_out = extra_channels[i] if i < len(extra_channels) else c_in
			block = [
				nn.Conv2d(
					c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False
				)
			]
			if extra_stage_use_bn:
				block.append(nn.BatchNorm2d(c_out))
			block.append(nn.ReLU(inplace=True))
			self.extra_down.append(nn.Sequential(*block))
			extra_out_channels.append(c_out)
			c_in = c_out

		# 3) ecs を最終確定：extra↓ を先頭に、pre↓ を末尾に
		ecs = (
			list(reversed(extra_out_channels)) if extra_out_channels else []
		) + ecs_base
		if self.pre_out_channels:
			ecs = ecs + self.pre_out_channels
		# Decoder
		self.decoder = UnetDecoder2d(
			encoder_channels=ecs,
			decoder_channels=decoder_channels,
			scale_factors=decoder_scales,
			upsample_mode=upsample_mode,
			attention_type=attention_type,
			intermediate_conv=intermediate_conv,
		)
		self.seg_head = SegmentationHead2d(
			in_channels=self.decoder.decoder_channels[-1],
			out_channels=out_chans,
		)

		# 推論時の TTA（flip）を使うか
		self.use_tta = True

	def _encode(self, x) -> list[torch.Tensor]:
		# ★各pre段の出力を控える
		pre_feats = []
		for b in self.pre_down:
			x = b(x)
			pre_feats.append(x)
			if getattr(self, 'print_shapes', False):
				print(f'[pre] {tuple(x.shape)}')

		# backbone → deepest-first
		feats = self.backbone(x)[::-1]

		# extra_down（最深側を前に積む）
		top = feats[0]
		for b in self.extra_down:
			top = b(top)
			feats = [top] + feats

		# ★pre_down 出力を浅い側（末尾）に積む
		feats = feats + pre_feats

		return feats

	@torch.inference_mode()
	def _proc_flip(self, x_in):
		x_flip = torch.flip(x_in, dims=[-2])

		feats = self._encode(x_flip)

		dec = self.decoder(feats)
		y = self.seg_head(dec[-1])
		y = torch.flip(y, dims=[-2])
		return y

	def forward(self, x):
		"""入力: x=(B,C,H,W)
		出力: y=(B,out_chans,H,W)  ※入力サイズに合わせて補間して返す
		"""
		H, W = x.shape[-2:]
		feats = self._encode(x)

		if getattr(self, 'print_shapes', False):
			for i, f in enumerate(feats):
				print(f'Encoder feature {i} shape:', f.shape)
		dec = self.decoder(feats)

		y = self.seg_head(dec[-1])  # 低解像度 → 後段で補間
		y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

		if self.training or not self.use_tta:
			return y

		# eval 時のみ簡易 TTA（左右反転）

		p1 = self._proc_flip(x)
		p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=False)
		return torch.quantile(torch.stack([y, p1]), q=0.5, dim=0)


if __name__ == '__main__':
	import torch

	# ダミー入力：バッチサイズ1、チャンネル数1、高さ128、幅6016
	dummy_input = torch.randn(1, 1, 128, 6016)

	# pre_stages=1 で横方向のみ 1/4 に縮小

	model = NetAE(
		backbone='caformer_b36.sail_in22k_ft_in1k',
		pretrained=True,
		stage_strides=[(2, 4), (2, 2), (2, 4), (2, 2)],
		pre_stages=2,
		pre_stage_strides=(
			(1, 1),
			(1, 2),
		),
	)
	adjust_first_conv_padding(model.backbone, padding=(3, 3))
	model.print_shapes = True
	model.eval()

	with torch.no_grad():
		output = model(dummy_input)
		print('Output shape:', output.shape)

# %%
import timm
import torch
import torch.nn.functional as F
from monai.networks.blocks import SubpixelUpsample, UpSample
from torch import nn

############################
# 汎用 Stem Adapter ユーティリティ
############################


def _find_first_conv(module: nn.Module):
	"""ツリーを先頭から走査して最初の Conv2d を返す（見つからなければ None）"""
	for m in module.modules():
		if isinstance(m, nn.Conv2d):
			return m
	return None


class PrePad2d(nn.Module):
	"""前段に挿入する可変パディング。mode='reflect' or 'zero'"""

	def __init__(self, pad=(0, 0, 0, 0), mode='reflect'):
		super().__init__()
		self.pad = pad
		if pad == (0, 0, 0, 0):
			self.op = nn.Identity()
		elif mode == 'reflect':
			self.op = nn.ReflectionPad2d(pad)
		elif mode == 'zero':
			# ZeroPad は Conv の padding で代替できるので Identity でも可
			self.op = nn.ZeroPad2d(pad)
		else:
			raise ValueError("mode must be 'reflect' or 'zero'")

	def forward(self, x):
		return self.op(x)


def adapt_stem(
	backbone: nn.Module,
	stride: tuple[int, int] | None = None,
	padding: tuple[int, int] | None = None,
	prepad: tuple[int, int, int, int] = (0, 0, 0, 0),
	prepad_mode: str = 'reflect',
):
	"""任意の timm バックボーンに対して、先頭 Conv の stride/padding と前段パディングを注入する。
	- `stride` / `padding` を None にすると既定値のまま（変更しない）
	- `prepad` は (left, right, top, bottom)
	- `stem` が無いモデルでも先頭 Conv を自動検出して変更
	"""
	# 1) 先頭 Conv を探す（優先: stem配下 → 全体）
	conv_target = None
	if hasattr(backbone, 'stem'):
		conv_target = _find_first_conv(backbone.stem)
	if conv_target is None:
		conv_target = _find_first_conv(backbone)
	if conv_target is None:
		raise RuntimeError('No Conv2d found in backbone to adapt.')

	# 2) stride/padding の上書き（指定があれば）
	if stride is not None:
		conv_target.stride = stride
	if padding is not None:
		conv_target.padding = padding

	# 3) 前段パディングの挿入
	if prepad != (0, 0, 0, 0):
		pad_block = PrePad2d(prepad, mode=prepad_mode)
		if hasattr(backbone, 'stem') and isinstance(backbone.stem, nn.Module):
			backbone.stem = nn.Sequential(pad_block, backbone.stem)
		else:
			# stem が無い場合は、モデル全体の先頭に差し込むためにラップ
			backbone.forward = _wrap_with_prepad(backbone.forward, pad_block)


def _wrap_with_prepad(forward_fn, pad_block):
	def new_forward(x, *args, **kwargs):
		x = pad_block(x)
		return forward_fn(x, *args, **kwargs)

	return new_forward


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
	def __init__(
		self,
		in_channels,
		skip_channels,
		out_channels,
		norm_layer: nn.Module = nn.Identity,
		attention_type: str = None,
		intermediate_conv: bool = False,
		upsample_mode: str = 'deconv',
		scale_factor: int = 2,
	):
		super().__init__()
		if upsample_mode == 'pixelshuffle':
			self.upsample = SubpixelUpsample(2, in_channels, scale_factor=scale_factor)
		else:
			self.upsample = UpSample(
				2,
				in_channels,
				out_channels=in_channels,
				scale_factor=scale_factor,
				mode=upsample_mode,
			)

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
		x = self.upsample(x)
		if self.intermediate_conv is not None:
			if skip is not None:
				skip = self.intermediate_conv(skip)
			else:
				x = self.intermediate_conv(x)
		if skip is not None:
			x = self.attention1(torch.cat([x, skip], dim=1))
		x = self.conv2(self.conv1(x))
		return self.attention2(x)


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
		upsample_mode: str = 'pixelshuffle',
	):
		super().__init__()
		if len(encoder_channels) == 4:
			decoder_channels = decoder_channels[1:]
		self.decoder_channels = decoder_channels
		if skip_channels is None:
			skip_channels = list(encoder_channels[1:]) + [0]

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
					scale_factors[i],
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
	def __init__(
		self,
		backbone: str,
		in_chans: int = 1,
		out_chans: int = 1,
		pretrained: bool = True,
		# stem 調整（任意）
		stem_stride: tuple[int, int] | None = None,  # 例: (4,1)
		stem_padding: tuple[int, int] | None = None,  # 例: (2,3)
		prepad: tuple[int, int, int, int] = (0, 0, 0, 0),  # 例: (1,1,20,20)
		prepad_mode: str = 'reflect',
		# decoder オプション
		decoder_channels: tuple = (256, 128, 64, 32),
		decoder_scales: tuple = (2, 2, 2, 2),
		upsample_mode: str = 'pixelshuffle',
		attention_type: str = 'scse',
		intermediate_conv: bool = True,
	):
		super().__init__()
		# Encoder (timm features_only)
		self.backbone = timm.create_model(
			backbone,
			in_chans=in_chans,
			pretrained=pretrained,
			features_only=True,
			drop_path_rate=0.0,
		)
		# 可搬な stem 調整
		adapt_stem(
			self.backbone,
			stride=stem_stride,
			padding=stem_padding,
			prepad=prepad,
			prepad_mode=prepad_mode,
		)

		# エンコーダ出力チャネル（深い順で来るので後で逆順）
		ecs = [fi['num_chs'] for fi in self.backbone.feature_info][::-1]

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

	@torch.inference_mode()
	def _proc_flip(self, x_in):
		x_flip = torch.flip(x_in, dims=[-2])
		feats = self.backbone(x_flip)[::-1]
		dec = self.decoder(feats)
		y = self.seg_head(dec[-1])
		y = torch.flip(y, dims=[-2])
		return y

	def forward(self, x):
		"""入力: x=(B,C,H,W)
		出力: y=(B,out_chans,H,W)  ※入力サイズに合わせて補間して返す
		"""
		H, W = x.shape[-2:]
		feats = self.backbone(x)[::-1]
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

	# ダミー入力：バッチサイズ1、チャンネル数5、高さ256、幅70
	dummy_input = torch.randn(1, 1, 128, 6016)

	# モデルの初期化（例: ConvNeXt-Tiny）
	model = NetAE(backbone='caformer_b36.sail_in22k_ft_in1k', pretrained=False)
	max_red = max(fi['reduction'] for fi in model.backbone.feature_info)
	print('Max reduction factor:', max_red)
	# 推論モード
	model.eval()

	# 順伝播テスト
	with torch.no_grad():
		output = model(dummy_input)
		print('Output shape:', output.shape)
		print('Output min:', output.min().item(), 'max:', output.max().item())

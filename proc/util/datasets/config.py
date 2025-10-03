from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoaderConfig:
	target_len: int  # 最終の時間長
	pad_traces_to: int = 128  # トレース本数の下側ゼロパディング上限


@dataclass(frozen=True)
class TraceSubsetSamplerConfig:
	# キー選択
	primary_keys: tuple[str, ...] | None = None  # 例: ('ffid','chno','cmp')
	primary_key_weights: tuple[float, ...] | None = None  # 重み（同順）
	# superwindow
	use_superwindow: bool = False
	sw_halfspan: int = 0  # 片側キー数（K = 1 + 2*sw_halfspan）
	sw_prob: float = 0.3  # superwindow を適用する確率（<1.0なら確率適用）
	# secondary 整列ルールを固定したいときに valid=True（従来互換）
	valid: bool = False
	# 連続サブセット本数（不足時は後段でパディング）
	subset_traces: int = 128

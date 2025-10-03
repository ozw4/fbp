from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoaderConfig:
	target_len: int  # 最終の時間長
	pad_traces_to: int = 128  # トレース本数の下側ゼロパディング上限

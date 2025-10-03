import random

import numpy as np

from .config import TraceSubsetSamplerConfig


class TraceSubsetSampler:
	"""ギャザー(=keyで定義されるトレース集合)から、連続サブセット(最大H=subset_traces)の
	trace indices を抽出するコンポーネント。

	入力: file_infos の 1 要素(dict) を想定（MaskedSegyGather が作る構造）
	  必須キー: '{ffid|chno|cmp}_unique_keys', '{ffid|chno|cmp}_key_to_indices',
	            '{ffid|chno|cmp}_values', 'offsets',
	            'ffid_centroids','chno_centroids' (あれば距離KNNに使用)
	出力: dict
	  - indices: np.ndarray[int64]    連続サブセットのインデックス
	  - pad_len: int                  subset_traces に満たない場合の不足本数
	  - key_name: str                 選んだ primary key 名('ffid'|'chno'|'cmp')
	  - secondary_key: str            二次整列に用いたキー('ffid'|'chno'|'offset')
	  - did_super: bool               このサンプルでsuperwindowを実際に適用したか
	  - primary_unique: str           サブセット内 primary 値のユニークを昇順に','連結
	"""

	def __init__(self, cfg: TraceSubsetSamplerConfig):
		self.cfg = cfg
		self._valid_primary = {'ffid', 'chno', 'cmp'}

	# ---- public API ----
	def draw(self, info: dict, *, py_random: random.Random | None = None) -> dict:
		r = py_random or random

		cmp_available = (
			bool(info.get('cmp_unique_keys'))
			and isinstance(info['cmp_unique_keys'], (list, tuple))
			and len(info['cmp_unique_keys']) > 0
		)

		# 1) primary key 候補の構築（重み付き / fallback 互換）
		key_candidates, weight_candidates = self._build_primary_candidates(
			cmp_available
		)

		# 選択
		if any(w > 0 for w in weight_candidates) and len(weight_candidates) == len(
			key_candidates
		):
			key_name = r.choices(key_candidates, weights=weight_candidates, k=1)[0]
		else:
			key_name = r.choice(key_candidates)

		unique_keys = info[f'{key_name}_unique_keys']
		key_to_indices = info[f'{key_name}_key_to_indices']
		if not unique_keys:
			raise RuntimeError(f'No unique keys for {key_name}')

		key = r.choice(unique_keys)
		indices = key_to_indices[key]

		# 2) superwindow（距離KNN／インデックスwindow fallback）
		apply_super, did_super = False, False
		if self.cfg.use_superwindow and self.cfg.sw_halfspan > 0:
			apply_super = True
			if float(self.cfg.sw_prob) < 1.0 and r.random() >= float(self.cfg.sw_prob):
				apply_super = False

			if apply_super:
				did_super = True
				indices = self._apply_superwindow(info, key_name, key, indices)

		indices = np.asarray(indices, dtype=np.int64)

		# 3) secondary 整列（従来条件の踏襲）
		secondary_key = self._choose_secondary_key(
			key_name, apply_super, self.cfg.valid, r
		)
		indices = self._stable_lexsort(info, key_name, secondary_key, indices)

		# 4) 連続サブセット抽出（最大 subset_traces）
		n_total = len(indices)
		H = int(self.cfg.subset_traces)
		if n_total >= H:
			start_idx = r.randint(0, n_total - H)
			subset_indices = indices[start_idx : start_idx + H]
			pad_len = 0
		else:
			subset_indices = indices
			pad_len = H - n_total

		subset_indices = np.asarray(subset_indices, dtype=np.int64)

		# 5) ログ用 primary ユニーク
		prim_vals_sel = info[f'{key_name}_values'][subset_indices].astype(np.int64)
		primary_label_values = np.unique(prim_vals_sel)
		primary_unique_str = ','.join(map(str, primary_label_values.tolist()))

		return dict(
			indices=subset_indices,
			pad_len=pad_len,
			key_name=key_name,
			secondary_key=secondary_key,
			did_super=did_super,
			primary_unique=primary_unique_str,
		)

	# ---- helpers ----
	def _build_primary_candidates(self, cmp_available: bool):
		if self.cfg.primary_keys:
			kc, wc = [], []
			for i, k in enumerate(self.cfg.primary_keys):
				if k not in self._valid_primary:
					continue
				if k == 'cmp' and not cmp_available:
					continue
				kc.append(k)
				if self.cfg.primary_key_weights and i < len(
					self.cfg.primary_key_weights
				):
					wc.append(max(float(self.cfg.primary_key_weights[i]), 0.0))
				else:
					wc.append(1.0)
			if not kc:  # fallback
				kc = ['ffid', 'chno'] + (['cmp'] if cmp_available else [])
				wc = [1.0] * len(kc)
		else:
			kc = ['ffid', 'chno'] + (['cmp'] if cmp_available else [])
			wc = [1.0] * len(kc)
		return kc, wc

	def _apply_superwindow(
		self, info: dict, key_name: str, key: int, indices: np.ndarray
	) -> np.ndarray:
		K = 1 + 2 * int(self.cfg.sw_halfspan)

		def _index_window():
			uniq = info.get(f'{key_name}_unique_keys', None)
			uniq_arr = (
				np.asarray(uniq, dtype=np.int64)
				if isinstance(uniq, (list, tuple))
				else np.asarray([], dtype=np.int64)
			)
			if uniq_arr.size > 0:
				uniq_sorted = np.sort(uniq_arr)
				center = int(key)
				pos = np.searchsorted(uniq_sorted, center)
				lo = max(0, pos - self.cfg.sw_halfspan)
				hi = min(len(uniq_sorted), pos + self.cfg.sw_halfspan + 1)
				return [int(k) for k in uniq_sorted[lo:hi]]
			return [int(key)]

		if key_name == 'ffid':
			cent = info.get('ffid_centroids')
			if isinstance(cent, dict) and int(key) in cent:
				keys = np.fromiter(cent.keys(), dtype=np.int64)
				coords = np.array([cent[int(k)] for k in keys], dtype=np.float64)
				cx, cy = cent[int(key)]
				d = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
				order = np.argsort(d)
				sel_keys = keys[order][:K]
				k2map = info['ffid_key_to_indices']
			else:
				sel_keys = np.asarray(_index_window(), dtype=np.int64)
				k2map = info[f'{key_name}_key_to_indices']
		elif key_name == 'chno':
			cent = info.get('chno_centroids')
			if isinstance(cent, dict) and int(key) in cent:
				keys = np.fromiter(cent.keys(), dtype=np.int64)
				coords = np.array([cent[int(k)] for k in keys], dtype=np.float64)
				cx, cy = cent[int(key)]
				d = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy)
				order = np.argsort(d)
				sel_keys = keys[order][:K]
				k2map = info['chno_key_to_indices']
			else:
				sel_keys = np.asarray(_index_window(), dtype=np.int64)
				k2map = info[f'{key_name}_key_to_indices']
		else:
			sel_keys = np.asarray(_index_window(), dtype=np.int64)
			k2map = info[f'{key_name}_key_to_indices']

		chunks = []
		for k2 in sel_keys:
			idxs = k2map.get(int(k2))
			if idxs is not None and len(idxs) > 0:
				chunks.append(idxs)
		if chunks:
			return np.concatenate(chunks).astype(np.int64)
		return np.asarray(indices, dtype=np.int64)

	def _choose_secondary_key(
		self, key_name: str, apply_super: bool, valid: bool, r: random.Random
	) -> str:
		if not apply_super and not valid:
			if key_name == 'ffid':
				return r.choice(('chno', 'offset'))
			if key_name == 'chno':
				return r.choice(('ffid', 'offset'))
			# 'cmp'
			return 'offset'
		if apply_super and not valid:
			return 'offset'
		if key_name == 'ffid':
			return 'chno'
		if key_name == 'chno':
			return 'ffid'
		return 'offset'

	def _stable_lexsort(
		self, info: dict, key_name: str, secondary: str, indices: np.ndarray
	) -> np.ndarray:
		prim_vals = info[f'{key_name}_values'][indices]
		if secondary == 'chno':
			sec_vals = info['chno_values'][indices]
		elif secondary == 'ffid':
			sec_vals = info['ffid_values'][indices]
		else:
			sec_vals = info['offsets'][indices]

		o = np.argsort(prim_vals, kind='mergesort')
		indices = indices[o]
		sec_vals = sec_vals[o]
		o2 = np.argsort(sec_vals, kind='mergesort')
		return indices[o2]

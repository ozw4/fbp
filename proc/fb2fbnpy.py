# %%
from pathlib import Path

import numpy as np
import segyio


def load_fb_irasformat(fb_file, dt):
	with open(fb_file) as f:
		lines = f.readlines()

	fb_values = []
	record_count = 0
	max_channel = 0

	for line in lines:
		line = line.rstrip()

		# レコードカウント
		if line.startswith('* rec.no.='):
			record_count += 1

		# fb 行処理
		elif line.startswith('fb'):
			payload = line[10:]

			# 5文字ずつに分割
			tokens = [payload[i : i + 5].strip() for i in range(0, len(payload), 5)]

			# チャンネル番号の最大値更新（2番目の値＝index 1）
			if len(tokens) >= 2:
				try:
					ch = int(tokens[1])
					max_channel = max(max_channel, ch)
				except ValueError:
					pass  # 数値変換失敗時は無視

			# 3カラム目以降が初動値
			for tok in tokens[2:]:
				try:
					val = int(tok)
					if val == -9999:
						val = 0
				except ValueError:
					val = 0
				fb_values.append(val)

	# NumPy 変換と前処理
	fb_array = np.array(fb_values)
	fb_array = fb_array // dt
	fb_array[fb_array < 0] = 0

	print(f'推定チャンネル数: {max_channel}')
	print(f'レコード数: {record_count}')
	assert len(fb_array) == record_count * max_channel, (
		f'レコード数とチャンネル数の不一致: {len(fb_array)} != {record_count} * {max_channel}'
	)
	return fb_array


data_dir = Path('/home/dcuser/data/ActiveSeisField')
field_dir_list = list(data_dir.glob('*'))
fb_files = list(data_dir.glob('*/*.crd'))

maxnt = 0
for field_dir in field_dir_list:
	print(field_dir)
	segy_file = list(field_dir.glob('*.sgy'))
	fb_file = list(field_dir.glob('*.crd'))

	if len(segy_file) != 1 or len(fb_file) != 1:
		print('Error: No or multiple files found in', field_dir)
		continue
	with segyio.open(segy_file[0], 'r', ignore_geometry=True) as f:
		dt = f.bin[segyio.BinField.Interval] / 1e4
		nt = f.samples.size
	maxnt = max(maxnt, nt)
	print(dt, nt)
	fb = load_fb_irasformat(fb_file[0], dt)
	np.save(fb_file[0].with_suffix('.npy'), fb)

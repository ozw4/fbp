from copy import deepcopy

import torch
from torch import nn


class ModelEMA(nn.Module):
	def __init__(self, model, decay=0.99, device=None):
		super().__init__()
		self.module = deepcopy(model)
		self.module.eval()
		self.decay = decay
		self.device = device
		if self.device is not None:
			self.module.to(device=device)

	def _update(self, model, update_fn):
		with torch.no_grad():
			for ema_v, model_v in zip(
				self.module.state_dict().values(),
				model.state_dict().values(),
				strict=False,
			):
				if self.device is not None:
					model_v = model_v.to(device=self.device)
				ema_v.copy_(update_fn(ema_v, model_v))

	def update(self, model):
		self._update(
			model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
		)

	def set(self, model):
		self._update(model, update_fn=lambda e, m: m)


class EnsembleModel(nn.Module):
	def __init__(self, models):
		super().__init__()
		self.models = nn.ModuleList(models).eval()

	def forward(self, x):
		output = None

		for m in self.models:
			logits = m(x)

			if output is None:
				output = logits
			else:
				output += logits

		output /= len(self.models)
		return output

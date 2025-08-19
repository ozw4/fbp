import random

import numpy as np
import torch

__all__ = ['worker_init_fn']

def worker_init_fn(worker_id: int) -> None:
	"""Seed numpy and random for each worker."""
	seed = torch.initial_seed() % 2**32
	np.random.seed(seed + worker_id)
	random.seed(seed + worker_id)

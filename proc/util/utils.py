# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import datetime
import errno
import itertools
import math
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


def set_seed(seed: int = 42):
	"""全ライブラリの乱数シードを固定して再現性を担保する。

	Args:
	    seed (int): 任意のシード値（デフォルト: 42）

	"""
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0, last_epoch=-1):
		self.warmup_steps = warmup_steps
		self.total_steps = total_steps
		self.eta_min = eta_min
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		step = self.last_epoch + 1
		if step < self.warmup_steps:
			return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
		if step <= self.total_steps:
			decay_step = step - self.warmup_steps
			decay_total = self.total_steps - self.warmup_steps
			return [
				self.eta_min
				+ (base_lr - self.eta_min)
				* 0.5
				* (1 + math.cos(math.pi * decay_step / decay_total))
				for base_lr in self.base_lrs
			]
		return [self.eta_min for _ in self.base_lrs]


class SmoothedValue:
	"""Track a series of values and provide access to smoothed values over a
	window or the global series average.
	"""

	def __init__(self, window_size=20, fmt=None):
		if fmt is None:
			fmt = '{median:.4f} ({global_avg:.4f})'
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0
		self.fmt = fmt

	def update(self, value, n=1):
		self.deque.append(value)
		self.count += n
		self.total += value * n

	def synchronize_between_processes(self):
		"""Warning: does not synchronize the deque!"""
		if not is_dist_avail_and_initialized():
			return
		t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
		dist.barrier()
		dist.all_reduce(t)
		t = t.tolist()
		self.count = int(t[0])
		self.total = t[1]

	@property
	def median(self):
		d = torch.tensor(list(self.deque))
		return d.median().item()

	@property
	def avg(self):
		d = torch.tensor(list(self.deque), dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self):
		return self.total / self.count

	@property
	def max(self):
		return max(self.deque)

	@property
	def value(self):
		return self.deque[-1]

	def __str__(self):
		return self.fmt.format(
			median=self.median,
			avg=self.avg,
			global_avg=self.global_avg,
			max=self.max,
			value=self.value,
		)


class MetricLogger:
	def __init__(self, delimiter='\t'):
		self.meters = defaultdict(SmoothedValue)
		self.delimiter = delimiter

	def update(self, **kwargs):
		for k, v in kwargs.items():
			if isinstance(v, torch.Tensor):
				v = v.item()
			assert isinstance(v, (float, int))
			self.meters[k].update(v)

	def __getattr__(self, attr):
		if attr in self.meters:
			return self.meters[attr]
		if attr in self.__dict__:
			return self.__dict__[attr]
		raise AttributeError(
			f"'{type(self).__name__}' object has no attribute '{attr}'"
		)

	def __str__(self):
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(f'{name}: {meter!s}')
		return self.delimiter.join(loss_str)

	def synchronize_between_processes(self):
		for meter in self.meters.values():
			meter.synchronize_between_processes()

	def add_meter(self, name, meter):
		self.meters[name] = meter

	def log_every(self, iterable, print_freq, header=None):
		if isinstance(iterable, list):
			length = max(len(x) for x in iterable)
			iterable = [x if len(x) == length else itertools.cycle(x) for x in iterable]
			iterable = zip(*iterable, strict=False)
		else:
			length = len(iterable)
		i = 0
		if not header:
			header = ''
		start_time = time.time()
		end = time.time()
		iter_time = SmoothedValue(fmt='{avg:.4f}')
		data_time = SmoothedValue(fmt='{avg:.4f}')
		space_fmt = ':' + str(len(str(length))) + 'd'
		if torch.cuda.is_available():
			log_msg = self.delimiter.join(
				[
					header,
					'[{0' + space_fmt + '}/{1}]',
					'eta: {eta}',
					'{meters}',
					'time: {time}',
					'data: {data}',
					'max mem: {memory:.0f}',
				]
			)
		else:
			log_msg = self.delimiter.join(
				[
					header,
					'[{0' + space_fmt + '}/{1}]',
					'eta: {eta}',
					'{meters}',
					'time: {time}',
					'data: {data}',
				]
			)
		MB = 1024.0 * 1024.0
		for obj in iterable:
			data_time.update(time.time() - end)
			yield obj  # <-- yield the batch in for loop
			iter_time.update(time.time() - end)
			if i % print_freq == 0:
				eta_seconds = iter_time.global_avg * (length - i)
				eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
				if torch.cuda.is_available():
					print(
						log_msg.format(
							i,
							length,
							eta=eta_string,
							meters=str(self),
							time=str(iter_time),
							data=str(data_time),
							memory=torch.cuda.max_memory_allocated() / MB,
						)
					)
				else:
					print(
						log_msg.format(
							i,
							length,
							eta=eta_string,
							meters=str(self),
							time=str(iter_time),
							data=str(data_time),
						)
					)
			i += 1
			end = time.time()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print(f'{header} Total time: {total_time_str}')


# Implemented according to H-PGNN, not useful
class NMSELoss(nn.Module):
	def __init__(self):
		super(NMSELoss, self).__init__()

	def forward(self, pred, gt):
		return torch.mean(
			((pred - gt) / (torch.amax(gt, (-2, -1), keepdim=True) + 1e-5)) ** 2
		)


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target[None])

		res = []
		for k in topk:
			correct_k = correct[:k].flatten().sum(dtype=torch.float32)
			res.append(correct_k * (100.0 / batch_size))
		return res


def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


def setup_for_distributed(is_master):
	"""This function disables printing when not in master process"""
	import builtins as __builtin__

	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		if is_master or force:
			builtin_print(*args, **kwargs)

	__builtin__.print = print


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_main_process():
	return get_rank() == 0


def save_on_master(*args, **kwargs):
	if is_main_process():
		torch.save(*args, **kwargs)


def init_distributed_mode(args):
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		args.rank = int(os.environ['RANK'])
		args.world_size = int(os.environ['WORLD_SIZE'])
		args.local_rank = int(os.environ['LOCAL_RANK'])
	elif 'SLURM_PROCID' in os.environ and args.world_size > 1:
		args.rank = int(os.environ['SLURM_PROCID'])
		args.local_rank = args.rank % torch.cuda.device_count()
	elif hasattr(args, 'rank'):
		pass
	else:
		print('Not using distributed mode')
		args.distributed = False
		return

	args.distributed = True

	torch.cuda.set_device(args.local_rank)
	args.dist_backend = 'nccl'
	print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
	torch.distributed.init_process_group(
		backend=args.dist_backend,
		init_method=args.dist_url,
		world_size=args.world_size,
		rank=args.rank,
	)
	setup_for_distributed(args.rank == 0)


def cal_psnr(gt, data, max_value):
	mse = np.mean((gt - data) ** 2)
	if mse == 0:
		return 100
	return 20 * np.log10(max_value / np.sqrt(mse))

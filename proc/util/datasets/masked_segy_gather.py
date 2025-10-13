import warnings

from .segy_gather_pipeline_dataset import SegyGatherPipelineDataset


class MaskedSegyGather(SegyGatherPipelineDataset):
	"""DEPRECATED: use SegyGatherPipelineDataset instead."""

	def __init__(self, *args, **kwargs):
		warnings.warn(
			'MaskedSegyGather is deprecated; use SegyGatherPipelineDataset.',
			DeprecationWarning,
			stacklevel=2,
		)
		super().__init__(*args, **kwargs)

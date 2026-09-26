from typing import Any, TypeAlias

import torch

LossDict: TypeAlias = dict[str, torch.Tensor]
LossMetrics: TypeAlias = dict[str, Any]

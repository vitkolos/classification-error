import torch

import classerr.optimization
import classerr.utils


def single(
    metadata: classerr.utils.Metadata,
    model_orig: torch.nn.Module,
    model_aprx: torch.nn.Module,
    point: torch.Tensor,
    target: int,
):
    point_b = point.unsqueeze(0)  # single input -> batch (to support any PyTorch network)
    result = classerr.optimization.analyze_linear_region(metadata, model_orig, model_aprx, point, target)
    return result

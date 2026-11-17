"""CPU-only metadata and bounded exact snapshots, never exported arrays."""
import torch
from .contract import UnsupportedOperation


def named_parameters(model):
    return dict(model.named_parameters(remove_duplicate=False))


def validate_parameter(parameter):
    if parameter.device.type != 'cpu' or parameter.dtype != torch.float32 or parameter.layout != torch.strided:
        raise UnsupportedOperation('only CPU dense FP32 parameters are supported')
    if not parameter.is_contiguous():
        raise UnsupportedOperation('non-contiguous parameters are outside v0.1 scope')


def gradient_kind(parameter):
    grad = parameter.grad
    if grad is None:
        return 'none'
    if grad.layout != torch.strided or grad.device.type != 'cpu' or grad.dtype != torch.float32:
        raise UnsupportedOperation('only CPU dense FP32 gradients are supported')
    if not bool(torch.isfinite(grad).all()):
        return 'nonfinite'
    return 'zero' if not bool(torch.count_nonzero(grad)) else 'nonzero'


def unique_parameters(named):
    result = {}
    for name, parameter in named.items():
        validate_parameter(parameter)
        if id(parameter) not in result:
            result[id(parameter)] = {'parameter': parameter, 'names': []}
        result[id(parameter)]['names'].append(name)
    ranges = []
    for key, record in result.items():
        p = record['parameter']
        if p.numel():
            start = p.data_ptr()
            ranges.append((start, start + p.numel() * p.element_size(), key))
    last_end = 0
    for start, end, _ in sorted(ranges):
        if start < last_end:
            raise UnsupportedOperation('distinct Parameter objects with overlapping memory ranges are unsupported')
        last_end = max(last_end, end)
    return result

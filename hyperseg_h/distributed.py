"""torchrun lifecycle, precision policy and whole-model DDP helpers."""
from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool = False

    @property
    def is_primary(self):
        return self.rank == 0


def init_distributed(action, device):
    """Only training creates a group. CPU torchrun uses Gloo for smoke tests."""
    keys = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')
    present = [key in os.environ for key in keys]
    if any(present) and not all(present):
        raise ValueError('torchrun requires RANK, LOCAL_RANK and WORLD_SIZE together')
    rank, local_rank, world_size = (
        (int(os.environ[key]) for key in keys) if all(present) else (0, 0, 1))
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError('Invalid torchrun rank/world size')
    device = torch.device(device)
    if action != 'train' or not all(present):
        return DistributedContext(rank, local_rank, world_size, device)
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    elif device.type != 'cpu':
        raise ValueError('DDP supports CUDA/NCCL or CPU/Gloo')
    dist.init_process_group(
        backend='nccl' if device.type == 'cuda' else 'gloo', init_method='env://',
        timeout=timedelta(minutes=30))
    return DistributedContext(rank, local_rank, world_size, device, True)


def cleanup_distributed(context):
    # Do not barrier in a finally block: a failed rank may no longer participate.
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_model(model, context):
    if not context.distributed:
        return model
    options = dict(find_unused_parameters=False, broadcast_buffers=False)
    if context.device.type == 'cuda':
        options.update(device_ids=[context.local_rank], output_device=context.local_rank)
    # Buffers stay rank-local (no SyncBatchNorm); frozen backbones remain eval.
    return DistributedDataParallel(model, **options)


def resolve_precision(precision=None, amp=False, device='cpu'):
    if precision is None:
        precision = ('fp16' if torch.device(device).type == 'cuda' else 'bf16') if amp else 'none'
    if precision not in ('bf16', 'fp16', 'none'):
        raise ValueError('precision must be bf16, fp16 or none')
    return precision


def autocast_dtype(precision):
    return {'bf16': torch.bfloat16, 'fp16': torch.float16, 'none': None}[precision]


def mean_metrics(metrics, context):
    if not context.distributed:
        return metrics
    values = torch.tensor(list(metrics.values()), dtype=torch.float64, device=context.device)
    dist.all_reduce(values)
    return dict(zip(metrics, (values / context.world_size).tolist()))

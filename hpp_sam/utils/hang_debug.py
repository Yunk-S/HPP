"""训练死锁排查：与 train.py 共用同一套 ``[HANG-DEBUG]`` 格式与 batch 限流。"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import torch

HangDebugInfo = Optional[Mapping[str, Any]]


def forward_hang_probe(info: HangDebugInfo, stage: str, extra: str = "") -> None:
    """在模型内部打点；``info`` 为 None 时不输出。

    ``info`` 通常由训练脚本构造，字段：
    - ``level`` (int): 0 关闭；1 仅前 ``first_batches`` 个 batch（``batch_idx`` 有效时）；2 每个 batch。
    - ``batch_idx`` (int): 当前 batch 下标；-1 表示不按 batch 限流（如 epoch_start）。
    - ``first_batches`` (int): 与 level 1 配合。
    - ``cuda_sync`` (bool): True 时探针前 ``torch.cuda.synchronize()``（极慢，用于区分 CPU 等 GPU）。
    """
    if info is None:
        return
    try:
        level = int(info.get("level", 0) or 0)
    except (TypeError, ValueError):
        level = 0
    if level <= 0:
        return
    try:
        batch_idx = int(info.get("batch_idx", -1) or -1)
    except (TypeError, ValueError):
        batch_idx = -1
    try:
        first_batches = int(info.get("first_batches", 5) or 5)
    except (TypeError, ValueError):
        first_batches = 5
    cuda_sync = bool(info.get("cuda_sync", False))
    if batch_idx >= 0 and level == 1:
        if batch_idx >= first_batches:
            return
    rank, world = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world = torch.distributed.get_world_size()
    t = time.monotonic()
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    msg = f"[HANG-DEBUG][rank{rank}/{world}][t={t:.3f}] {stage}"
    if extra:
        msg += f" | {extra}"
    if cuda_sync:
        msg += " | post_cuda_sync"
    print(msg, flush=True)

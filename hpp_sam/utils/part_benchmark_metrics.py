"""
Part / Point interactive segmentation benchmark metrics (Point-SAM, PartNeXt style).

Convention (matches common tables in Point-SAM, PartSAM-class papers):
    IoU@K  —  segmentation IoU **after the K-th user click** (prompt iteration K),
              i.e. using the mask produced at prompt round K.

Implementation:
    We store prompt rounds as 0-based indices ``i_iter`` in the training code.
    Therefore:  IoU@K  corresponds to  ``i_iter == K - 1``.

    mIoU     —  mean of IoU@1, IoU@3, IoU@5, IoU@7, IoU@10 using ``numpy.nanmean``:
              if ``prompt_iters`` is too small, missing clicks contribute NaN and
              are excluded from the mean (documented in logs).

References:
    - Point-SAM: Promptable 3D Segmentation (interactive clicks)
    - PartNeXt / PartNet-style part segmentation benchmarks with multi-click IoU curves
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Union

import numpy as np

# Clicks used in standard comparison tables (1-based click count)
BENCHMARK_CLICKS: tuple[int, ...] = (1, 3, 5, 7, 10)


def click_to_iter(click: int) -> int:
    """Map 1-based click index to 0-based prompt iteration index."""
    return int(click) - 1


def compute_benchmark_from_aux(
    aux: List[Mapping[str, Any]],
) -> Dict[str, float]:
    """
    Training step: ``aux`` is the list from Criterion (one entry per prompt iteration).

    Returns keys: IoU@1, IoU@3, ..., IoU@10, mIoU
    """
    out: Dict[str, float] = {}
    for click in BENCHMARK_CLICKS:
        idx = click_to_iter(click)
        if idx < len(aux) and "iou" in aux[idx]:
            iou_t = aux[idx]["iou"]
            # [B*M] tensor
            val = float(iou_t.detach().float().mean().cpu().item())
            out[f"IoU@{click}"] = val
        else:
            out[f"IoU@{click}"] = float("nan")

    bench = np.array([out[f"IoU@{c}"] for c in BENCHMARK_CLICKS], dtype=np.float64)
    out["mIoU"] = float(np.nanmean(bench))
    return out


def compute_benchmark_from_epoch_ious(
    epoch_ious: Mapping[Union[int, str], List[float]],
) -> Dict[str, float]:
    """
    Validation epoch: ``epoch_ious`` maps iteration index -> list of per-sample IoUs.
    Integer keys are prompt iterations; ``\"best\"`` (multimask @ iter 0) is ignored for IoU@K.
    """
    out: Dict[str, float] = {}
    for click in BENCHMARK_CLICKS:
        idx = click_to_iter(click)
        if idx in epoch_ious and len(epoch_ious[idx]) > 0:
            out[f"IoU@{click}"] = float(np.mean(epoch_ious[idx]))
        else:
            out[f"IoU@{click}"] = float("nan")

    bench = np.array([out[f"IoU@{c}"] for c in BENCHMARK_CLICKS], dtype=np.float64)
    out["mIoU"] = float(np.nanmean(bench))
    return out


def format_benchmark_for_postfix(
    bench: Mapping[str, float],
    max_len: int = 200,
) -> str:
    """Short single-line string for tqdm (may truncate)."""
    parts = []
    for c in BENCHMARK_CLICKS:
        key = f"IoU@{c}"
        v = bench.get(key, float("nan"))
        if np.isnan(v):
            parts.append(f"@{c}=nan")
        else:
            parts.append(f"@{c}={v:.3f}")
    m = bench.get("mIoU", float("nan"))
    m_s = "nan" if np.isnan(m) else f"{m:.3f}"
    s = " ".join(parts) + f" mIoU={m_s}"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def print_benchmark_metric_help() -> None:
    print(
        "[metrics] Benchmark keys: IoU@1, IoU@3, IoU@5, IoU@7, IoU@10 (after K-th click), "
        "mIoU = nan-mean of those five. Requires prompt_iters >= 10 for IoU@10."
    )

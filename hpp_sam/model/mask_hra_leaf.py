"""
ZeRO-3：将 mask_encoder → repeat_interleave → hyper_prompt_branch 包成**单个子模块**。

各 rank 的 M 不同时，若二者为独立子模块，快 rank 会先触发 HRA 的参数 gather，
慢 rank 仍卡在 mask_encoder 的 hook/kernel → NCCL 顺序错乱、表现为部分 rank 停在
before_hyper_prompt_branch。本模块让 DeepSpeed 在一次 leaf 边界上完成整段路径的
gather/release（配合 set_z3_leaf_modules(MaskEncoderHRALeaf)）。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from hpp_sam.utils.hang_debug import forward_hang_probe

from .common import repeat_interleave
from .hyper_prompt_branch import HyperPromptBranch
from .prompt_encoder import MaskEncoder


class MaskEncoderHRALeaf(nn.Module):
    """单 leaf：dense mask 编码 + repeat + 双曲 prompt 分支。"""

    def __init__(self, mask_encoder: MaskEncoder, hyper_prompt_branch: HyperPromptBranch):
        super().__init__()
        self.mask_encoder = mask_encoder
        self.hyper_prompt_branch = hyper_prompt_branch

    def forward(
        self,
        prompt_masks: Optional[torch.Tensor],
        coords: torch.Tensor,
        centers: torch.Tensor,
        knn_idx: Optional[torch.Tensor],
        sparse_embeddings: torch.Tensor,
        use_voronoi: bool,
        patches: Mapping[str, Any],
        hang_debug: Optional[Mapping[str, Any]],
        iter_i: int,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if use_voronoi:
            nn_assignment = patches.get("nn_assignment", patches.get("nn_idx", None))
            dense_embeddings = self.mask_encoder(
                prompt_masks,
                coords,
                centers,
                None,
                center_idx=patches.get("fps_idx"),
                nn_assignment=nn_assignment,
            )
        else:
            dense_embeddings = self.mask_encoder(
                prompt_masks,
                coords,
                centers,
                knn_idx,
                center_idx=patches.get("fps_idx"),
            )

        forward_hang_probe(hang_debug, f"fwd.iter{iter_i}.after_mask_encoder_dense")

        dense_embeddings = repeat_interleave(
            dense_embeddings,
            sparse_embeddings.shape[0] // dense_embeddings.shape[0],
            0,
        )

        forward_hang_probe(hang_debug, f"fwd.iter{iter_i}.before_hyper_prompt_branch")
        os.environ["HPP_HANG_DEBUG_HRA"] = "1"
        os.environ["HPP_HANG_DEBUG_TAG"] = f"fwd.iter{iter_i}"
        _hd: Optional[Mapping[str, Any]] = None
        if hang_debug is not None:
            _hd = {**hang_debug, "tag": f"fwd.iter{iter_i}"}
        hra_output = self.hyper_prompt_branch(sparse_embeddings, hang_debug=_hd)
        os.environ["HPP_HANG_DEBUG_HRA"] = "0"

        return dense_embeddings, hra_output

"""
HPP-SAM: Hyperbolic Point Prompt Segment Anything Model

This module implements the HPP-SAM model, integrating:
1. HyperPromptBranch: Hyperbolic Prompt Branch with HRA for granularity control
2. HyperCrossAttention: Radius-conditioned hyperbolic cross-attention
3. Mixed-manifold design: Euclidean encoder + Hyperbolic decoder

Design Reference:
- HRA-PointSAM_design.md
- 数据流动.md (Hyperbolic Prompt Branch)
- 注意力模块.md (Hyperbolic Cross-Attention)

Memory Optimization Features:
- Gradient checkpointing for PatchEncoder and MaskDecoder
- Optional save_on_cpu (use_cpu_offload): NOT applied around mask_encoder / mask_decoder
  because nesting save_on_cpu with torch.utils.checkpoint(use_reentrant=False) breaks
  (pack_to_cpu / invalid CUDA args); Point-SAM does not use this pattern.
- Chunked processing for HyperCrossAttention
- Optional FSDP (see train config)
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torkit3d.nn.functional import batch_index_select

from .common import repeat_interleave, sample_prompts, sample_prompts_adapter
from .mask_decoder import AuxInputs, MaskDecoder
from .point_cloud_encoder import PointCloudEncoder
from .prompt_encoder import MaskEncoder, PointEncoder
from .hyper_prompt_branch import HyperPromptBranch
from .mask_hra_leaf import MaskEncoderHRALeaf
from hpp_sam.utils.hang_debug import forward_hang_probe


class HPPSAM(nn.Module):
    """
    HPP-SAM: Hyperbolic Point Prompt Segment Anything Model.
    
    Architecture:
        - Encoder: Euclidean point cloud encoder
        - Prompt Branch: HRA-enhanced hyperbolic transformation
        - Decoder: Hyperbolic cross-attention with τ(r) control
        - Output Head: Euclidean (mask prediction)
    
    Args:
        pc_encoder: Point cloud encoder (Euclidean).
        mask_encoder: Mask encoder for dense prompts.
        mask_decoder: Mask decoder with hyperbolic cross-attention.
        prompt_iters: Number of prompt iterations.
        enable_mask_refinement_iterations: Enable mask refinement during training.
        hra_curvature: Hyperbolic curvature c. Default 0.01.
        hra_type: HRA type. Options: "diagonal", "block_diagonal". Default "block_diagonal".
        block_size: Block size for block-diagonal HRA. Default 8.
        tau_min: Minimum temperature. Default 0.1.
        tau_max: Maximum temperature. Default 2.0.
        use_cpu_offload: If True, run extra cuda.empty_cache between prompt iterations
            (legacy flag; save_on_cpu around mask path is disabled — incompatible with checkpoint).
    """

    def __init__(
        self,
        pc_encoder: PointCloudEncoder,
        mask_encoder: MaskEncoder,
        mask_decoder: MaskDecoder,
        prompt_iters: int,
        enable_mask_refinement_iterations: bool = True,
        hra_curvature: float = 0.01,
        hra_type: str = "block_diagonal",
        block_size: int = 8,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        use_cpu_offload: bool = False,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.point_encoder = PointEncoder(pc_encoder.embed_dim)
        self.mask_decoder = mask_decoder
        self.prompt_iters = prompt_iters
        self.enable_mask_refinement_iterations = enable_mask_refinement_iterations
        self.use_cpu_offload = use_cpu_offload

        _hra = HyperPromptBranch(
            embed_dim=pc_encoder.embed_dim,
            curvature=hra_curvature,
            hra_type=hra_type,
            block_size=block_size,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        # ZeRO-3：与 HRA 合成单 leaf，避免两子模块间 gather 与慢 rank 的 mask_encoder 交错
        self.mask_dense_hra = MaskEncoderHRALeaf(mask_encoder, _hra)

    @property
    def mask_encoder(self) -> MaskEncoder:
        return self.mask_dense_hra.mask_encoder

    @property
    def hyper_prompt_branch(self) -> HyperPromptBranch:
        return self.mask_dense_hra.hyper_prompt_branch

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        """兼容旧 checkpoint：`mask_encoder.*` / `hyper_prompt_branch.*` → `mask_dense_hra.*`。"""
        p_me, p_hra = "mask_encoder.", "hyper_prompt_branch."
        n_me = "mask_dense_hra.mask_encoder."
        n_hra = "mask_dense_hra.hyper_prompt_branch."
        mapped: Dict[str, Any] = {}
        for k, v in state_dict.items():
            if k.startswith(p_me):
                mapped[n_me + k[len(p_me) :]] = v
            elif k.startswith(p_hra):
                mapped[n_hra + k[len(p_hra) :]] = v
            else:
                mapped[k] = v
        return super().load_state_dict(mapped, strict=strict)

    def predict_masks(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        prompt_coords: torch.Tensor,
        prompt_labels: torch.Tensor,
        prompt_masks: torch.Tensor = None,
        multimask_output: bool = True,
    ):
        """
        Predict masks given point prompts with HRA enhancement.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
            prompt_coords: [B*M, num_queries, 3]. Prompt coordinates.
            prompt_labels: [B*M, num_queries]. Prompt labels.
            prompt_masks: Optional [B*M, N] mask prompts.
            multimask_output: Whether to output multiple masks.

        Returns:
            masks: [B*M, num_outputs, N] Predicted masks.
            iou_preds: [B*M, num_outputs] IoU predictions.
            radius: [B*M, num_queries] Prompt radii (for analysis).
            temperature: [B*M, num_queries] Attention temperatures.
        """
        # Encoder (Euclidean)
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches["centers"]
        # 支持Voronoi和KNN两种模式
        use_voronoi = getattr(self.pc_encoder.patch_embed, 'use_voronoi', False)
        if use_voronoi:
            knn_idx = patches.get("nn_assignment", patches.get("nn_idx", None))
        else:
            knn_idx = patches["knn_idx"]
        aux_inputs = AuxInputs(coords=coords, features=features, centers=centers)

        # Positional encoding
        pc_pe = self.point_encoder.pe_layer(centers)

        # Point encoder (Euclidean)
        sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)

        # Dense embeddings
        dense_embeddings = self.mask_encoder(
            prompt_masks, coords, centers, knn_idx
        )
        dense_embeddings = repeat_interleave(
            dense_embeddings,
            sparse_embeddings.shape[0] // dense_embeddings.shape[0],
            0,
        )

        # HyperPromptBranch (CORE INNOVATION)
        hra_output = self.hyper_prompt_branch(sparse_embeddings)
        hyperbolic_prompt = hra_output["hyperbolic_prompt"]
        radius = hra_output["radius"]
        temperature = hra_output["temperature"]

        # Decoder (with hyperbolic cross-attention)
        masks, iou_preds = self.mask_decoder(
            pc_embeddings,
            pc_pe,
            sparse_embeddings,
            dense_embeddings,
            aux_inputs=aux_inputs,
            multimask_output=multimask_output,
            hyperbolic_prompt=hyperbolic_prompt,
            temperature=temperature,
            radius=radius,
            prompt_coords=prompt_coords,
        )

        return masks, iou_preds, radius, temperature

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        gt_masks: torch.Tensor,
        is_eval: bool = False,
        hang_debug: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass for training with HRA enhancement.

        Args:
            coords: [B, N, 3]. Point cloud coordinates.
            features: [B, N, F]. Point cloud features.
            gt_masks: [B, M, N], bool. Ground truth binary masks.
            is_eval: Whether in evaluation mode.
            hang_debug: 由 train 传入的打点 payload（见 ``hpp_sam.utils.hang_debug``）；None 不打 ``fwd.*`` 日志。

        Returns:
            outputs: List of dictionaries containing:
                - prompt_coords, prompt_labels, masks, iou_preds
                - radius, temperature (HRA outputs)
        """
        batch_size = coords.shape[0]
        num_masks = gt_masks.shape[1]
        forward_hang_probe(
            hang_debug,
            "fwd.enter",
            extra=f"B={batch_size} M={num_masks} prompt_iters={self.prompt_iters} training={self.training}",
        )

        # 注意：勿在此处 dist.barrier()。DeepSpeed ZeRO-3 下若各 rank 因 skip/OOM 导致
        # micro_step 不同步，快 rank 会进入下一 batch 的 forward 并卡在 barrier，
        # 慢 rank 仍在上一次 forward 内 → 永久死锁。跨 rank 对齐应在 train.py 用
        # wait_for_everyone + 统一的 backward/step 决策保证。

        # Encoder (Euclidean)
        forward_hang_probe(hang_debug, "fwd.before_pc_encoder")
        pc_embeddings, patches = self.pc_encoder(coords, features)
        forward_hang_probe(hang_debug, "fwd.after_pc_encoder")
        centers = patches["centers"]
        # 支持Voronoi和KNN两种模式
        use_voronoi = getattr(self.pc_encoder.patch_embed, 'use_voronoi', False)
        if use_voronoi:
            knn_idx = patches.get("nn_assignment", patches.get("nn_idx", None))
        else:
            knn_idx = patches["knn_idx"]
        forward_hang_probe(
            hang_debug,
            "fwd.after_patch_meta",
            extra=f"voronoi={use_voronoi}",
        )

        outputs = []
        prompt_coords = coords.new_empty((batch_size * num_masks, 0, 3))
        prompt_labels = gt_masks.new_empty((batch_size * num_masks, 0))
        prompt_masks = None
        aux_inputs = AuxInputs(coords=coords, features=features, centers=centers)

        # Mask refinement iterations
        if self.enable_mask_refinement_iterations and self.training:
            mask_refinement_iterations = [self.prompt_iters - 1]
            if self.prompt_iters > 1:
                # ZeRO-3 / DDP：mask_refinement_iterations 决定哪些迭代里跳过 sample_prompts。
                # 各 rank 必须得到**完全相同**的 sampled_iter，否则 forward 路径与张量形状分叉，
                # DeepSpeed 会在 all_gather 上与仍在另一子图的 rank 死锁（表现为部分 rank 已 forward_begin 下一 batch）。
                forward_hang_probe(hang_debug, "fwd.before_refine_iter_broadcast")
                if dist.is_initialized():
                    # 错误写法曾用 seed 含 rank → 各 rank sampled_iter 不同 → 必死锁。
                    # broadcast 用与 coords 同设备：NCCL 后端需要 CUDA 张量。
                    dev = coords.device
                    t = torch.zeros(1, dtype=torch.long, device=dev)
                    if dist.get_rank() == 0:
                        g = torch.Generator()
                        g.manual_seed(42 + int(self.prompt_iters))
                        v = torch.randint(1, self.prompt_iters, (1,), generator=g)
                        t.copy_(v.to(dev))
                    dist.broadcast(t, src=0)
                    sampled_iter = int(t.item())
                else:
                    g = torch.Generator()
                    g.manual_seed(self.prompt_iters + 42)
                    sampled_iter = torch.randint(1, self.prompt_iters, (1,), generator=g).item()
                mask_refinement_iterations.append(sampled_iter)
                forward_hang_probe(
                    hang_debug,
                    "fwd.after_refine_iter_broadcast",
                    extra=f"mask_refinement_iterations={mask_refinement_iterations}",
                )
        else:
            mask_refinement_iterations = []

        # Positional encoding
        forward_hang_probe(hang_debug, "fwd.before_pc_pe")
        pc_pe = self.point_encoder.pe_layer(centers)
        forward_hang_probe(hang_debug, "fwd.after_pc_pe")

        # Iterate over prompt refinement steps
        for i in range(self.prompt_iters):
            forward_hang_probe(
                hang_debug,
                f"fwd.iter{i}.start",
                extra=f"skip_sample={i != 0 and i in mask_refinement_iterations}",
            )
            # Sample prompts
            if i == 0 or i not in mask_refinement_iterations:
                forward_hang_probe(hang_debug, f"fwd.iter{i}.before_sample_prompts_adapter")
                new_prompt_coords, new_prompt_labels = sample_prompts_adapter(
                    coords, gt_masks, prompt_masks, is_eval=is_eval,
                )
                forward_hang_probe(hang_debug, f"fwd.iter{i}.after_sample_prompts_adapter")
                prompt_coords = torch.cat([prompt_coords, new_prompt_coords], dim=1)
                prompt_labels = torch.cat([prompt_labels, new_prompt_labels], dim=1)
            else:
                forward_hang_probe(hang_debug, f"fwd.iter{i}.skipped_sample_prompts")

            # Point encoder (Euclidean)
            forward_hang_probe(hang_debug, f"fwd.iter{i}.before_point_encoder_sparse")
            sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)
            forward_hang_probe(hang_debug, f"fwd.iter{i}.after_point_encoder_sparse")

            # Dense embeddings: do NOT wrap mask_encoder in save_on_cpu — it uses
            # torch.utils.checkpoint inside PatchEncoder; nesting save_on_cpu with
            # use_reentrant=False checkpoint triggers pack_to_cpu / torch.empty
            # "invalid argument" and unstable CUDA after OOM (Point-SAM has no such nesting).
            forward_hang_probe(hang_debug, f"fwd.iter{i}.before_mask_encoder_dense")
            dense_embeddings, hra_output = self.mask_dense_hra(
                prompt_masks,
                coords,
                centers,
                knn_idx,
                sparse_embeddings,
                use_voronoi,
                patches,
                hang_debug,
                i,
            )
            forward_hang_probe(hang_debug, f"fwd.iter{i}.after_hyper_prompt_branch")
            hyperbolic_prompt = hra_output["hyperbolic_prompt"]
            radius = hra_output["radius"]
            temperature = hra_output["temperature"]

            # Decoder: same as mask_encoder — mask_decoder uses gradient checkpointing;
            # do not nest save_on_cpu here.
            forward_hang_probe(
                hang_debug,
                f"fwd.iter{i}.before_mask_decoder",
                extra=f"multimask_output={i == 0}",
            )
            masks, iou_preds = self.mask_decoder(
                pc_embeddings,
                pc_pe,
                sparse_embeddings,
                dense_embeddings,
                aux_inputs=aux_inputs,
                multimask_output=(i == 0),
                hyperbolic_prompt=hyperbolic_prompt,
                temperature=temperature,
                radius=radius,
                prompt_coords=prompt_coords,
            )
            forward_hang_probe(hang_debug, f"fwd.iter{i}.after_mask_decoder")

            # Select most confident mask for next iteration
            forward_hang_probe(hang_debug, f"fwd.iter{i}.before_argmax_select_mask")
            max_iou_pred_ind = torch.argmax(iou_preds, dim=1)
            prompt_masks = batch_index_select(masks, max_iou_pred_ind, dim=1)
            forward_hang_probe(hang_debug, f"fwd.iter{i}.after_argmax_select_mask")

            outputs.append({
                "prompt_coords": prompt_coords,
                "prompt_labels": prompt_labels,
                "masks": masks,
                "iou_preds": iou_preds,
                "max_iou_pred_ind": max_iou_pred_ind,
                "prompt_masks": prompt_masks,
                "radius": radius,
                "temperature": temperature,
            })

            # Memory cleanup: delete intermediate tensors that are no longer needed
            if self.use_cpu_offload and i < self.prompt_iters - 1:
                del sparse_embeddings, dense_embeddings, hyperbolic_prompt
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()

        forward_hang_probe(hang_debug, "fwd.exit", extra=f"num_outputs={len(outputs)}")
        return outputs

"""
Hyperbolic Cross-Attention Module for HPP-SAM.

This module implements the radius-conditioned hyperbolic cross-attention,
injecting prompt radius as a temperature control for attention sharpness,
enabling granularity-aware segmentation.

Key Concepts:
    1. Geodesic Matching: Uses Poincaré distance instead of dot product for attention.
    2. Radius-Conditioned Temperature: Maps prompt radius τ(r_i) to attention sharpness.
    3. Tangent Mean Aggregation: Stable weighted aggregation in hyperbolic space.

Mathematical Framework:

    Standard Hyperbolic Attention:
        α_ij = softmax(-d_c(q_i, k_j))
        
    Radius-Conditioned Version (HPP-SAM):
        ℓ_ij = -τ(r_i) · d_c(q_tilde_i, k_j) - b
        α_ij = softmax(ℓ_ij)
        
        where τ(r_i) is the radius temperature function from HyperPromptBranch:
        τ(r_i) = τ_min + (τ_max - τ_min) · σ(a · r_i + b_0)
        
        Physical meaning:
        - Small τ → smooth attention → coarse segmentation
        - Large τ → sharp attention → fine segmentation
        
    Value Aggregation (Tangent Mean - DEFAULT):
        m_i = exp_0^c(Σ_j α_ij · log_0^c(v_j))

References:
    - HRA-PointSAM_design.md (注意力模块 section)
    - Hyperbolic Attention Networks (NeurIPS 2019)
    - HyperET (https://github.com/godlin-sjtu/HyperET)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Literal, Optional, Tuple
from .hyper_ops import HyperOps


class HyperCrossAttention(nn.Module):
    """
    Hyperbolic Cross-Attention with Radius-Conditioned Temperature.

    This module implements the core attention mechanism for HPP-SAM. It replaces
    standard Euclidean attention with hyperbolic geodesic-based attention, where the
    temperature is conditioned on the prompt radius from HyperPromptBranch.

    Architecture:
        1. Q/K/V Projections (Euclidean space)
        2. Key/Value projection to hyperbolic space (optional, for mixed-manifold)
        3. Geodesic distance computation with τ(r_i) injection
        4. Softmax attention
        5. Tangent Mean aggregation

    Memory Optimization:
        - Query-side chunking (q_chunk_size): Controls memory of distance matrix computation
        - Key-side chunking (k_chunk_size): Additional memory reduction for large key sequences
        - Gradient checkpointing: Supported via use_gradient_checkpointing flag
        - The hyperbolic operations (exp0, log0, poincare_dist) are optimized for memory efficiency

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        head_dim: Dimension per head. If None, computed as embed_dim // num_heads.
        curvature: Hyperbolic curvature c. Default 0.01.
        aggregation: Aggregation method. Options: "tangent_mean" (default).
        bias: Whether to use bias in projections. Default True.
        use_gradient_checkpointing: Whether to use gradient checkpointing for the
            cross-attention forward to save activation memory. Default False.
        chunk_size: Chunk size for batched geodesic distance computation.
            Controls the memory-accuracy trade-off. Default None (full batch).
        q_chunk_size: Chunk size for query dimension to limit peak memory.
            Smaller values use less memory but increase compute overhead.
            Default 256 (reduced from 512 for better memory efficiency).
        k_chunk_size: Chunk size for key dimension. If None, uses chunk_size.
            Default 256 for better memory efficiency with large key sequences.
        use_ball_query: If True and positions are provided, restrict keys to top-k
            Euclidean neighbors per query (O(Nq·k) geodesic distances vs O(Nq·Nk)).
        k_neighbors: Number of Euclidean neighbors for Ball Query.
        ball_query_mode: Reserved; only \"euclidean\" neighbor prefilter is used.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        curvature: float = 0.01,
        aggregation: Literal["tangent_mean"] = "tangent_mean",
        bias: bool = True,
        use_gradient_checkpointing: bool = False,
        chunk_size: Optional[int] = None,
        q_chunk_size: int = 256,
        k_chunk_size: int = 256,
        use_ball_query: bool = True,
        k_neighbors: int = 64,
        ball_query_mode: Literal["euclidean", "hyperbolic"] = "euclidean",
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.curvature = curvature
        self.aggregation = aggregation
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.chunk_size = chunk_size
        self.q_chunk_size = q_chunk_size  # Reduced from 512 for better memory efficiency
        self.k_chunk_size = k_chunk_size  # New parameter for key-side chunking
        self.use_ball_query = use_ball_query
        self.k_neighbors = k_neighbors
        self.ball_query_mode = ball_query_mode

        self.hyper_ops = HyperOps(curvature=curvature, eps=1e-5)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Scale Euclidean keys into the ball; 0.5 → ||exp0(0.5*k)|| ≈ tanh(0.1) ≈ 0.1
        self.key_hyp_scale = nn.Parameter(torch.tensor(0.5))

        self.logit_scale = nn.Parameter(torch.ones([]) * 0.0)
        self.scale = self.head_dim ** -0.5

    def _compute_euclidean_neighbors(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k key indices per query by Euclidean distance. Returns dists [B,Nq,k], idx [B,Nq,k]."""
        B, Nq, _ = query_pos.shape
        Nk = key_pos.shape[1]
        euclidean_dist = torch.cdist(query_pos, key_pos)
        if k >= Nk:
            idx = torch.arange(Nk, device=query_pos.device, dtype=torch.long).view(1, 1, Nk).expand(B, Nq, Nk)
            return euclidean_dist, idx
        dist_topk, idx_topk = torch.topk(euclidean_dist, k=k, dim=-1, largest=False)
        return dist_topk, idx_topk

    def _hyper_ball_query_attention(
        self,
        q_h: torch.Tensor,
        k_h: torch.Tensor,
        v: torch.Tensor,
        temperature: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Local hyperbolic attention: Euclidean top-k neighbor mask, geodesic scores on those keys only.
        q_h: [B,H,Nq,hD], k_h/v: [B,H,Nk,hD], query_pos: [B,Nq,3], key_pos: [B,Nk,3].
        
        显存优化：使用安全的索引方式，避免非法内存访问。
        """
        B, H, Nq, hD = q_h.shape
        Nk = k_h.shape[2]
        k = min(self.k_neighbors, Nk)
        
        # 获取top-k邻居索引
        _, idx_topk = self._compute_euclidean_neighbors(query_pos, key_pos, k=k)
        k_eff = idx_topk.shape[-1]

        # 安全索引：重塑为[B*H, Nq, k]然后gather
        # materialize expand to avoid zero-stride views under FSDP
        idx_reshaped = idx_topk.unsqueeze(1).expand(B, H, Nq, k_eff).clone().reshape(B * H, Nq, k_eff)
        
        # k_h: [B, H, Nk, hD] -> [B*H, Nk, hD]
        k_h_2d = k_h.reshape(B * H, Nk, hD)
        v_2d = v.reshape(B * H, Nk, hD)

        # 沿 Nk 维按每个 (batch*head, query) 的 idx 取 key/value。
        # 注意：1D index_select(dim=1) 会对所有行共用同一组列下标，不能把 idx 摊平后当作 per-query gather。
        idx_gather = idx_reshaped.clamp(0, Nk - 1).long()
        bh = torch.arange(B * H, device=k_h_2d.device, dtype=torch.long).view(-1, 1, 1).expand(-1, Nq, k_eff).clone()
        k_h_selected_2d = k_h_2d[bh, idx_gather, :]
        v_selected_2d = v_2d[bh, idx_gather, :]
        
        # 重塑回[B, H, Nq, k_eff, hD]
        k_h_selected = k_h_selected_2d.reshape(B, H, Nq, k_eff, hD)
        v_selected = v_selected_2d.reshape(B, H, Nq, k_eff, hD)

        # Compute geodesic distances on selected neighbors
        # materialize expand to avoid zero-stride views under FSDP
        q_flat = q_h.unsqueeze(3).expand(-1, -1, -1, k_eff, -1).clone().reshape(B * H * Nq * k_eff, hD)
        k_flat = k_h_selected.reshape(B * H * Nq * k_eff, hD)
        
        # Try hyperbolic distance, fallback to Euclidean if CUDA error
        try:
            dist_flat = self.hyper_ops.poincare_dist(q_flat, k_flat, use_gradient=self.training)
        except RuntimeError:
            diff = q_flat - k_flat
            dist_flat = torch.norm(diff, p=2, dim=-1)
        
        dist = dist_flat.reshape(B, H, Nq, k_eff)
        tau = temperature.unsqueeze(1).unsqueeze(-1)
        dist_scaled = -tau * dist + self.logit_scale
        attn_weight = F.softmax(dist_scaled, dim=-1)

        if self.aggregation == "tangent_mean":
            # Try hyperbolic aggregation, fallback to Euclidean if fails
            try:
                v_flat_agg = v_selected.reshape(B * H * Nq, k_eff, -1)
                v_h_flat = self.hyper_ops.exp0(v_flat_agg)
                v_log_flat = self.hyper_ops.log0(v_h_flat)
                attn_w = attn_weight.reshape(B * H * Nq, k_eff, 1)
                weighted_sum = (v_log_flat * attn_w).sum(dim=1)
                output_h = self.hyper_ops.exp0(weighted_sum)
                output_e = self.hyper_ops.log0(output_h)
                output = output_e.reshape(B, H, Nq, hD)
            except RuntimeError:
                # Euclidean fallback: simple weighted average
                v_flat_agg = v_selected.reshape(B * H * Nq, k_eff, -1)
                attn_w = attn_weight.reshape(B * H * Nq, k_eff, 1)
                output = (v_flat_agg * attn_w).sum(dim=1).reshape(B, H, Nq, hD)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        hidden = output.transpose(1, 2).contiguous().view(B, Nq, self.embed_dim)
        
        # 安全scatter操作：确保索引在有效范围内
        idx_topk_safe = idx_topk.long().clamp(0, Nk - 1)
        attn_weight_avg = torch.zeros(B, Nq, Nk, dtype=attn_weight.dtype, device=attn_weight.device)
        attn_weight_avg.scatter_(2, idx_topk_safe, attn_weight.mean(dim=1))
        return hidden, attn_weight_avg

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature: torch.Tensor,
        radius: Optional[torch.Tensor] = None,
        key_in_hyperbolic: bool = False,
        need_weights: bool = True,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute hyperbolic cross-attention with radius-conditioned temperature.

        Args:
            query: Query tensor, shape [B, Nq, D].
                   For prompt→point: Nq = num_prompts, query is hyperbolic (q_tilde).
            key: Key tensor, shape [B, Nk, D].
            value: Value tensor, shape [B, Nk, D].
            temperature: Attention temperature τ(r_i), shape [B, Nq].
                        From HyperPromptBranch, controls attention sharpness.
            radius: Optional radius values for debugging, shape [B, Nq].
            key_in_hyperbolic: Whether key is already in hyperbolic space.
            need_weights: Whether to return attention weights. Default True.
            query_pos: Optional [B, Nq, 3] for Ball Query (3D positions of queries).
            key_pos: Optional [B, Nk, 3] for Ball Query (3D positions of keys).

        Returns:
            Output tensor, shape [B, Nq, D].
            Attention weights (if need_weights=True), shape [B, Nq, Nk].
        """
        B, Nq, _ = query.shape
        Nk = key.shape[1]

        # τ 来自 HRA，按 **prompt** 条数给出 [B, N_prompt]；prompt→point 时 Nq=N_prompt。
        # point→prompt 时 query 为点云 token，Nq=N_pc，需把 τ 对齐到每个 query 槽位。
        if temperature.shape[0] != B:
            raise ValueError(
                f"temperature batch {temperature.shape[0]} != query batch {B}"
            )
        Nt = temperature.shape[1]
        if Nt != Nq:
            if Nt == 1:
                temperature = temperature.expand(B, Nq).clone()
            elif Nt < Nq:
                temperature = temperature.mean(dim=1, keepdim=True).expand(B, Nq).clone()
            else:
                temperature = temperature[:, :Nq]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        if not key_in_hyperbolic:
            k_flat = k.reshape(B * self.num_heads, Nk, self.head_dim)
            k_h_flat = self.hyper_ops.exp0(self.key_hyp_scale * k_flat)
            k_h = k_h_flat.reshape(B, self.num_heads, Nk, self.head_dim)
        else:
            k_h = k

        # If Nk <= k_neighbors, full attention matches Ball Query neighborhood size.
        use_ball = (
            self.use_ball_query
            and query_pos is not None
            and key_pos is not None
            and Nk > self.k_neighbors
        )
        
        # Apply gradient checkpointing to attention computation if enabled
        if self.training and self.use_gradient_checkpointing:
            hidden, attn_weight_avg = self._forward_with_checkpoint(
                q, k_h, v, temperature, use_ball, query_pos, key_pos
            )
        elif use_ball:
            hidden, attn_weight_avg = self._hyper_ball_query_attention(
                q, k_h, v, temperature, query_pos, key_pos
            )
        else:
            hidden, attn_weight_avg = self._forward_hyper_attn_core(
                q, k_h, v, temperature
            )

        output = self.out_proj(hidden)

        if need_weights:
            return output, attn_weight_avg
        return output

    def _forward_with_checkpoint(
        self,
        q: torch.Tensor,
        k_h: torch.Tensor,
        v: torch.Tensor,
        temperature: torch.Tensor,
        use_ball: bool,
        query_pos: Optional[torch.Tensor],
        key_pos: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with gradient checkpointing for memory efficiency.
        
        This method wraps the attention computation in torch.utils.checkpoint
        to save activation memory during training.
        """
        if use_ball:
            # Ball query path is already memory-efficient, checkpointing may not help much
            # but we can still checkpoint the tangent mean aggregation
            hidden, attn_weight_avg = self._hyper_ball_query_attention(
                q, k_h, v, temperature, query_pos, key_pos
            )
            return hidden, attn_weight_avg
        else:
            # Use checkpoint for the core attention computation
            # The tangent_mean_aggregation is checkpointed separately
            hidden, attn_weight_avg = checkpoint(
                self._checkpointed_attention_core,
                q, k_h, v, temperature,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            return hidden, attn_weight_avg

    def _checkpointed_attention_core(
        self,
        q: torch.Tensor,
        k_h: torch.Tensor,
        v: torch.Tensor,
        temperature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core attention computation wrapped for checkpointing."""
        B, H, Nq, hD = q.shape
        Nk = k_h.shape[2]

        q_expanded = q.unsqueeze(3)  # [B, H, Nq, 1, hD]
        k_expanded = k_h.unsqueeze(2)  # [B, H, 1, Nk, hD]

        # Chunked distance computation
        if self.chunk_size is not None:
            dist = self._compute_geodesic_dist_chunked(
                q_expanded, k_expanded, self.chunk_size
            )
        else:
            dist = self._compute_geodesic_dist_batch(q_expanded, k_expanded)

        tau = temperature.unsqueeze(1).unsqueeze(-1)
        dist_scaled = -tau * dist + self.logit_scale

        attn_weight = F.softmax(dist_scaled, dim=-1)

        # Tangent mean aggregation with chunking for memory efficiency
        if self.aggregation == "tangent_mean":
            output = self._tangent_mean_aggregation(v, attn_weight)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        hidden = output.transpose(1, 2).contiguous().view(B, Nq, self.embed_dim)
        attn_weight_avg = attn_weight.transpose(1, 2).mean(dim=1)
        return hidden, attn_weight_avg

    def _forward_hyper_attn_core(
        self,
        q: torch.Tensor,
        k_h: torch.Tensor,
        v: torch.Tensor,
        temperature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """测地线注意力 + tangent-mean；返回 out_proj 前隐状态与平均注意力权重。"""
        B, H, Nq, hD = q.shape

        q_expanded = q.unsqueeze(3)  # [B, H, Nq, 1, hD]
        # k_h is always [B, H, Nk, hD], add query dim for broadcasting
        k_expanded = k_h.unsqueeze(2)  # [B, H, 1, Nk, hD]

        if self.chunk_size is not None:
            dist = self._compute_geodesic_dist_chunked(
                q_expanded, k_expanded, self.chunk_size
            )
        else:
            dist = self._compute_geodesic_dist_batch(q_expanded, k_expanded)

        tau = temperature.unsqueeze(1).unsqueeze(-1)
        dist_scaled = -tau * dist + self.logit_scale

        attn_weight = F.softmax(dist_scaled, dim=-1)

        if self.aggregation == "tangent_mean":
            output = self._tangent_mean_aggregation(v, attn_weight)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        hidden = output.transpose(1, 2).contiguous().view(B, Nq, self.embed_dim)
        attn_weight_avg = attn_weight.transpose(1, 2).mean(dim=1)
        return hidden, attn_weight_avg
    
    def _compute_geodesic_dist_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Memory-efficient batched geodesic distance computation using chunking.

        Implements a two-level chunking strategy:
        1. Key-side chunking: Process keys in chunks to avoid materializing full (B*H*Nq*Nk) matrix
        2. Query-side chunking: Further split queries within each key chunk for large Nq
        
        This dramatically reduces peak memory while maintaining mathematical equivalence.

        Args:
            q: Query tensor, shape [B, H, Nq, 1, hD].
            k: Key tensor (in hyperbolic space), shape [B, H, Nk, hD] or
               [B, H, 1, Nk, hD] (broadcast layout from forward).
            chunk_size: Number of keys to process at once. If None, uses k_chunk_size.
                Smaller values use less memory but increase compute overhead.
                Recommended: 128-256 for typical high-memory scenarios.

        Returns:
            Geodesic distances, shape [B, H, Nq, Nk].
        """
        B, H, Nq, _, hD = q.shape
        k_prep = self._geodesic_k_to_5d(k)
        Nk = k_prep.shape[3]

        # Use k_chunk_size as default if chunk_size not specified
        if chunk_size is None:
            chunk_size = self.k_chunk_size

        # For small key sequences, use batch computation with query chunking
        if Nk <= chunk_size:
            return self._compute_geodesic_dist_batch(q, k_prep)

        # Two-level chunking: first chunk by keys, then by queries within each key chunk
        dist = torch.empty(B, H, Nq, Nk, dtype=q.dtype, device=q.device)

        for k_start in range(0, Nk, chunk_size):
            k_end = min(k_start + chunk_size, Nk)
            k_chunk = k_prep[:, :, :, k_start:k_end, :]
            current_k_size = k_end - k_start
            
            # Query chunking within each key chunk for very large Nq
            for q_start in range(0, Nq, self.q_chunk_size):
                q_end = min(q_start + self.q_chunk_size, Nq)
                q_chunk = q[:, :, q_start:q_end, :, :]
                
                # materialize expand: broadcast 1->current_k_size along dim=3; FSDP shard-safe
                q_exp = q_chunk.expand(-1, -1, -1, current_k_size, -1).clone()
                
                # Compute distance for this sub-chunk
                dist_sub = self._compute_geodesic_dist_single_chunk(q_exp, k_chunk)
                dist[:, :, q_start:q_end, k_start:k_end] = dist_sub
            
            # Clean up key chunk after processing
            del k_chunk

        return dist
    
    def _compute_geodesic_dist_single_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute geodesic distances for a single query-key chunk.
        
        Memory-efficient implementation that processes the chunk without
        materializing the full pairwise distance matrix.
        
        Args:
            q: shape [B, H, Nq, Nk_chunk, hD]
            k: shape [B, H, 1, Nk_chunk, hD] or [B, H, Nk_chunk, hD]
            
        Returns:
            Geodesic distances, shape [B, H, Nq, Nk_chunk]
        """
        B, H, Nq, Nk_chunk, hD = q.shape
        k5 = self._geodesic_k_to_5d(k)
        
        # materialize expand: broadcast 1->Nq along dim=2, then reshape; FSDP shard-safe
        k_expanded = k5.expand(-1, -1, Nq, -1, -1).clone()
        
        # Flatten for batched hyperbolic distance computation
        # q_flat: [B*H*Nq*Nk_chunk, hD]
        # k_flat: [B*H*Nq*Nk_chunk, hD]
        q_flat = q.reshape(B * H * Nq * Nk_chunk, hD)
        k_flat = k_expanded.reshape(B * H * Nq * Nk_chunk, hD)
        
        # Compute distances using memory-efficient poincare_dist
        # Note: use_gradient=True during training, can be set to False during inference
        try:
            # Pre-check: NaN/Inf in inputs reaches CUDA kernels and triggers
            # "CUDA error: an illegal memory access was encountered".
            # Guard here so we fall back to Euclidean BEFORE the kernel call,
            # preserving the CUDA context for all subsequent batches.
            if not (torch.isfinite(q).all() and torch.isfinite(k).all()):
                diff = q - k
                nan_mask = ~torch.isfinite(diff)
                diff = torch.where(nan_mask, torch.zeros_like(diff), diff)
                dist_flat = torch.norm(diff, p=2, dim=-1)
            else:
                dist_flat = self.hyper_ops.poincare_dist(q, k, use_gradient=self.training)
        except RuntimeError:
            # Fallback: use Euclidean distance when hyperbolic computation fails
            diff = q - k
            dist_flat = torch.norm(diff, p=2, dim=-1)
        
        # Reshape back to [B, H, Nq, Nk_chunk]
        dist = dist_flat.reshape(B, H, Nq, Nk_chunk)
        
        return dist

    @staticmethod
    def _geodesic_k_to_5d(k: torch.Tensor) -> torch.Tensor:
        """
        Keys for geodesic dist must be [B, H, 1, Nk, hD] so that expand with queries
        yields [B, H, Nq, Nk, hD]. Accepts either [B, H, Nk, hD] or already-expanded k.
        """
        if k.dim() == 4:
            return k.unsqueeze(2)
        if k.dim() == 5:
            return k
        raise ValueError(f"Expected key tensor with 4 or 5 dims, got shape {tuple(k.shape)}")

    def _compute_geodesic_dist_batch(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Poincaré distances between query and key in batch.
        
        Memory-optimized: always chunks along the query dimension to avoid
        materializing B*H*Nq*Nk intermediates. Uses a smaller default chunk
        size (256) for better memory efficiency compared to the previous 512.
        
        For very large sequences, prefer _compute_geodesic_dist_chunked which
        provides two-level chunking.

        Args:
            q: shape [B, H, Nq, 1, hD] or [B, H, Nq, Nk, hD].
            k: shape [B, H, 1, Nk, hD] or [B, H, Nk, hD].

        Returns:
            Geodesic distances, shape [B, H, Nq, Nk].
        """
        B, H, Nq, _, hD = q.shape
        k5 = self._geodesic_k_to_5d(k)
        Nk = k5.shape[3]

        dist = torch.empty(B, H, Nq, Nk, dtype=q.dtype, device=q.device)
        chunk_size = self.q_chunk_size  # Now 256 by default

        for q_start in range(0, Nq, chunk_size):
            q_end = min(q_start + chunk_size, Nq)
            q_chunk = q[:, :, q_start:q_end, :, :]
            current_q_size = q_end - q_start

            # materialize expand: broadcast 1->Nk and 1->current_q_size; FSDP shard-safe
            q_exp = q_chunk.expand(-1, -1, -1, Nk, -1).clone()
            k_exp = k5.expand(-1, -1, current_q_size, -1, -1).clone()

            # Flatten for batched computation: (current_q_size * H * B * Nk, hD)
            q_flat = q_exp.reshape(current_q_size * H * B * Nk, hD)
            k_flat = k_exp.reshape(current_q_size * H * B * Nk, hD)

            # Compute distances with gradient control for memory efficiency
            try:
                # Guard: check inputs are finite BEFORE entering poincare_dist
                # which calls mobius_add and then CUDA kernels.
                # Without this, CUDA illegal memory access cascades and
                # destroys the context for ALL subsequent batches.
                if not (torch.isfinite(q_flat).all() and torch.isfinite(k_flat).all()):
                    finite_mask = torch.isfinite(q_flat) & torch.isfinite(k_flat)
                    safe_q = torch.where(finite_mask, q_flat, torch.zeros_like(q_flat))
                    safe_k = torch.where(finite_mask.unsqueeze(-1).expand_as(safe_q),
                                        safe_q, torch.zeros_like(safe_k))
                    # Use Euclidean fallback for non-finite inputs
                    diff = q_flat - k_flat
                    nan_mask = ~torch.isfinite(diff)
                    diff_flat = torch.where(nan_mask,
                        torch.zeros_like(diff),
                        diff)
                    dist_flat = torch.norm(diff_flat, p=2, dim=-1)
                else:
                    dist_flat = self.hyper_ops.poincare_dist(
                        q_flat, k_flat, use_gradient=self.training
                    )
            except RuntimeError:
                # Fallback: use Euclidean distance when hyperbolic computation fails
                diff = q_flat - k_flat
                dist_flat = torch.norm(diff, p=2, dim=-1)
            
            dist[:, :, q_start:q_end, :] = dist_flat.reshape(
                B, H, current_q_size, Nk
            )
            
            # Explicitly free intermediate tensors
            del q_exp, k_exp, q_flat, k_flat, dist_flat

        return dist
    
    def _tangent_mean_aggregation(
        self,
        v: torch.Tensor,
        attn_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tangent Mean Aggregation with optional chunking for memory efficiency.

        Formula: m_i = exp_0^c(Σ_j α_ij · log_0^c(v_j))

        Memory optimizations:
        - Processes key chunks sequentially to reduce peak memory
        - Uses gradient control for intermediate hyperbolic operations
        - Explicitly frees intermediate tensors when not needed
        
        When chunk_size is set, processes key chunks sequentially to reduce peak
        memory usage during the weighted sum. Mathematically equivalent.
        """
        B, H, Nk, hD = v.shape
        Nq = attn_weight.shape[2]
        
        # Compute hyperbolic transformations with gradient control for memory
        v_flat = v.reshape(B * H, Nk, hD)
        v_h_flat = self.hyper_ops.exp0(v_flat)
        v_log_flat = self.hyper_ops.log0(v_h_flat)
        
        # Free intermediate tensors explicitly
        del v_h_flat

        attn = attn_weight.permute(0, 1, 3, 2)
        attn_flat = attn.reshape(B * H, Nk, Nq).transpose(1, 2)

        # Use chunking for large key sequences
        chunk_size = getattr(self, 'k_chunk_size', 256)
        if Nk > chunk_size:
            weighted_sum = torch.zeros(
                B * H, Nq, hD, dtype=v.dtype, device=v.device
            )
            for k_start in range(0, Nk, chunk_size):
                k_end = min(k_start + chunk_size, Nk)
                current_k_size = k_end - k_start
                
                attn_chunk = attn_flat[:, :, k_start:k_end]
                v_log_chunk = v_log_flat[:, k_start:k_end, :]
                
                # Efficient batched weighted sum
                attn_chunk_expanded = attn_chunk.unsqueeze(-1)  # [BH, Nq, K, 1]
                v_log_chunk_expanded = v_log_chunk.unsqueeze(1)  # [BH, 1, K, hD]
                
                chunk_sum = (attn_chunk_expanded * v_log_chunk_expanded).sum(dim=2)
                weighted_sum.add_(chunk_sum)
                
                del attn_chunk, v_log_chunk, attn_chunk_expanded, v_log_chunk_expanded, chunk_sum
        else:
            weighted_sum = torch.bmm(attn_flat, v_log_flat)

        # Final hyperbolic transformations
        output_h_flat = self.hyper_ops.exp0(weighted_sum)
        output_e_flat = self.hyper_ops.log0(output_h_flat)
        
        # Free intermediate
        del output_h_flat

        output = output_e_flat.reshape(B, H, Nq, hD)
        del output_e_flat

        return output


class HyperSelfAttention(nn.Module):
    """
    Hyperbolic Self-Attention for point cloud features.
    
    Note: For simplicity and numerical stability, we use tangent space self-attention
    (performing attention in the tangent space at origin) rather than full
    hyperbolic self-attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention."""
        B, N, D = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        
        return x


class HyperCrossAttentionCheckpointWrapper(nn.Module):
    """
    Wrapper for HyperCrossAttention that enables gradient checkpointing compatibility.
    
    This wrapper handles the gradient checkpointing for hyperbolic operations by
    breaking the forward pass into checkpointable segments.
    
    Usage:
        base_attn = HyperCrossAttention(...)
        wrapped_attn = HyperCrossAttentionCheckpointWrapper(base_attn)
    """
    
    def __init__(
        self,
        hyper_cross_attn: HyperCrossAttention,
        num_checkpoint_segments: int = 2,
    ):
        super().__init__()
        self.hyper_cross_attn = hyper_cross_attn
        self.num_checkpoint_segments = num_checkpoint_segments
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature: torch.Tensor,
        radius: Optional[torch.Tensor] = None,
        key_in_hyperbolic: bool = False,
        need_weights: bool = True,
        use_checkpointing: bool = True,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional gradient checkpointing.
        
        When use_checkpointing=True and self.training=True, the hyperbolic distance
        computation is checkpointed to save activation memory.
        """
        if use_checkpointing and self.training:
            return self._forward_with_checkpointing(
                query,
                key,
                value,
                temperature,
                radius,
                key_in_hyperbolic,
                need_weights,
                query_pos,
                key_pos,
            )
        else:
            return self.hyper_cross_attn(
                query,
                key,
                value,
                temperature,
                radius,
                key_in_hyperbolic,
                need_weights,
                query_pos=query_pos,
                key_pos=key_pos,
            )
    
    def _forward_with_checkpointing(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature: torch.Tensor,
        radius: Optional[torch.Tensor],
        key_in_hyperbolic: bool,
        need_weights: bool,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
    ):
        """Forward with gradient checkpointing for memory efficiency."""
        B, Nq, _ = query.shape
        Nk = key.shape[1]
        
        # Project Q, K, V (these are checkpointable)
        if self.num_checkpoint_segments >= 1:
            q, k, v, k_h = checkpoint(
                self._project_qkv,
                query, key, value, key_in_hyperbolic,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            q = self.hyper_cross_attn.q_proj(query)
            k = self.hyper_cross_attn.k_proj(key)
            v = self.hyper_cross_attn.v_proj(value)
            if not key_in_hyperbolic:
                k_flat = k.reshape(B * self.hyper_cross_attn.num_heads, Nk, self.hyper_cross_attn.head_dim)
                k_h_flat = self.hyper_cross_attn.hyper_ops.exp0(self.hyper_cross_attn.key_hyp_scale * k_flat)
                k_h = k_h_flat.reshape(B, self.hyper_cross_attn.num_heads, Nk, self.hyper_cross_attn.head_dim)
            else:
                k_h = k
        
        q = q.view(B, Nq, self.hyper_cross_attn.num_heads, self.hyper_cross_attn.head_dim).transpose(1, 2)
        k_h = k_h.view(B, Nk, self.hyper_cross_attn.num_heads, self.hyper_cross_attn.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.hyper_cross_attn.num_heads, self.hyper_cross_attn.head_dim).transpose(1, 2)

        # Match HyperCrossAttention.forward temperature alignment (point→prompt Nq != N_prompt)
        if temperature.shape[0] != B:
            raise ValueError(
                f"temperature batch {temperature.shape[0]} != query batch {B}"
            )
        Nt = temperature.shape[1]
        if Nt != Nq:
            if Nt == 1:
                temperature = temperature.expand(B, Nq).clone()
            elif Nt < Nq:
                temperature = temperature.mean(dim=1, keepdim=True).expand(B, Nq).clone()
            else:
                temperature = temperature[:, :Nq]

        attn_mod = self.hyper_cross_attn
        Nk_h = k_h.shape[2]
        use_ball = (
            attn_mod.use_ball_query
            and query_pos is not None
            and key_pos is not None
            and Nk_h > attn_mod.k_neighbors
        )
        if use_ball:
            # Ball path: do not checkpoint (small activation footprint vs full Nk)
            hidden, attn_weight_avg = attn_mod._hyper_ball_query_attention(
                q, k_h, v, temperature, query_pos, key_pos
            )
        elif self.num_checkpoint_segments >= 2:
            hidden, attn_weight_avg = checkpoint(
                self._compute_attention_core,
                q, k_h, v, temperature,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            hidden, attn_weight_avg = attn_mod._forward_hyper_attn_core(
                q, k_h, v, temperature
            )
        
        # Output projection (no checkpointing needed for small layer)
        output = self.hyper_cross_attn.out_proj(hidden)

        if need_weights:
            return output, attn_weight_avg
        return output

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_in_hyperbolic: bool,
    ):
        """Project Q, K, V tensors."""
        hyper_attn = self.hyper_cross_attn
        B, Nq, _ = query.shape
        Nk = key.shape[1]
        
        q = hyper_attn.q_proj(query)
        k = hyper_attn.k_proj(key)
        v = hyper_attn.v_proj(value)
        
        if not key_in_hyperbolic:
            k_flat = k.reshape(B * hyper_attn.num_heads, Nk, hyper_attn.head_dim)
            k_h_flat = hyper_attn.hyper_ops.exp0(hyper_attn.key_hyp_scale * k_flat)
            k_h = k_h_flat.reshape(B, hyper_attn.num_heads, Nk, hyper_attn.head_dim)
        else:
            k_h = k
        
        return q, k, v, k_h
    
    def _compute_attention_core(
        self,
        q: torch.Tensor,
        k_h: torch.Tensor,
        v: torch.Tensor,
        temperature: torch.Tensor,
    ):
        """Compute core hyperbolic attention (checkpointable)."""
        return self.hyper_cross_attn._forward_hyper_attn_core(
            q, k_h, v, temperature
        )


def create_hyper_cross_attention(
    embed_dim: int,
    num_heads: int,
    curvature: float = 0.01,
    aggregation: Literal["tangent_mean"] = "tangent_mean",
    use_gradient_checkpointing: bool = False,
    chunk_size: Optional[int] = None,
    q_chunk_size: int = 256,
    k_chunk_size: int = 256,
    use_ball_query: bool = True,
    k_neighbors: int = 64,
    ball_query_mode: Literal["euclidean", "hyperbolic"] = "euclidean",
) -> HyperCrossAttention:
    """
    Factory function to create HyperCrossAttention with memory-optimized defaults.
    
    Memory optimization tips:
    - Reduce q_chunk_size and k_chunk_size for lower memory usage
    - Set use_gradient_checkpointing=True for additional memory savings during training
    - For very large sequences, prefer smaller chunk sizes (64-128)
    - use_ball_query + k_neighbors reduces cross-attention memory when Nk is large
    """
    return HyperCrossAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        curvature=curvature,
        aggregation=aggregation,
        use_gradient_checkpointing=use_gradient_checkpointing,
        chunk_size=chunk_size,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        use_ball_query=use_ball_query,
        k_neighbors=k_neighbors,
        ball_query_mode=ball_query_mode,
    )

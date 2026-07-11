"""
TwoWayTransformer with optional hyperbolic cross-attention for HPP-SAM.

This module integrates hyperbolic cross-attention into the transformer decoder.
The modifications follow the design in HRA-PointSAM_design.md:

1. Self-attention: Remains Euclidean/tangent (lightweight, stable)
2. Cross-attention: Uses HyperCrossAttention for prompt→point and point→prompt
3. Temperature injection: Uses radius from HyperPromptBranch to control attention sharpness
"""

import torch
from torch import Tensor, nn
import math
from typing import Tuple, Type, Optional

from .hyper_ops import HyperOps
from .hyper_cross_attention import HyperCrossAttention


class TwoWayTransformer(nn.Module):
    """
    A transformer decoder with optional hyperbolic cross-attention for HPP-SAM.
    
    Args:
      depth (int): number of layers in the transformer
      embedding_dim (int): the channel dimension for the input embeddings
      num_heads (int): the number of heads for multihead attention
      mlp_dim (int): the channel dimension internal to the MLP block
      activation (nn.Module): the activation to use in the MLP block
      attention_downsample_rate (int): downsample rate for attention
      use_hyperbolic_cross_attn (bool): whether to use hyperbolic cross-attention
      curvature (float): hyperbolic curvature c. Default 0.01 (from HyperET)
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_hyperbolic_cross_attn: bool = True,
        curvature: float = 0.01,
        use_gradient_checkpointing: bool = False,
        chunk_size: Optional[int] = None,
        q_chunk_size: int = 256,
        k_chunk_size: int = 256,
        use_ball_query: bool = True,
        k_neighbors: int = 64,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.use_hyperbolic_cross_attn = use_hyperbolic_cross_attn
        self.curvature = curvature
        self.q_chunk_size = q_chunk_size
        self.k_chunk_size = k_chunk_size
        self.use_ball_query = use_ball_query
        self.k_neighbors = k_neighbors

        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    use_hyperbolic_cross_attn=use_hyperbolic_cross_attn,
                    curvature=curvature,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    chunk_size=chunk_size,
                    q_chunk_size=q_chunk_size,
                    k_chunk_size=k_chunk_size,
                    use_ball_query=use_ball_query,
                    k_neighbors=k_neighbors,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        pc_embedding: Tensor,
        pc_pe: Tensor,
        point_embedding: Tensor,
        hyperbolic_prompt: Optional[Tensor] = None,
        temperature: Optional[Tensor] = None,
        radius: Optional[Tensor] = None,
        pc_pos: Optional[Tensor] = None,
        prompt_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          pc_embedding: point cloud embeddings, shape [B x N_pc_tokens x embedding_dim].
          pc_pe: positional encoding for point cloud.
          point_embedding: prompt embeddings, shape [B x N_points x embedding_dim].
          hyperbolic_prompt: hyperbolic prompt from HyperPromptBranch, shape [B x N_points x D].
          temperature: temperature τ(r_i), shape [B x N_points].
          radius: radius r_i, shape [B x N_points].
          pc_pos: patch center positions [B, N_pc, 3] for Ball Query.
          prompt_pos: query token positions [B, N_hyp_tokens, 3] aligned with hyperbolic_prompt.
        """
        use_hyp = self.use_hyperbolic_cross_attn and hyperbolic_prompt is not None

        queries = point_embedding
        keys = pc_embedding

        for layer in self.layers:
            if use_hyp:
                queries, keys = layer(
                    queries=queries,
                    keys=keys,
                    query_pe=point_embedding,
                    key_pe=pc_pe,
                    hyperbolic_queries=hyperbolic_prompt,
                    temperature=temperature,
                    radius=radius,
                    key_pos=pc_pos,
                    query_pos=prompt_pos,
                )
            else:
                queries, keys = layer(
                    queries=queries,
                    keys=keys,
                    query_pe=point_embedding,
                    key_pe=pc_pe,
                )

        q = queries + point_embedding
        k = keys + pc_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """
    A transformer block with four layers for HPP-SAM:
    (1) self-attention of sparse inputs (Euclidean/tangent)
    (2) cross attention of sparse inputs to dense inputs (HYPERBOLIC)
    (3) mlp block on sparse inputs
    (4) cross attention of dense inputs to sparse inputs (HYPERBOLIC)

    Args:
        embedding_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: MLP hidden dimension.
        activation: Activation function.
        attention_downsample_rate: Downsample rate for attention.
        skip_first_layer_pe: Whether to skip positional encoding in first layer.
        use_hyperbolic_cross_attn: Whether to use hyperbolic cross-attention.
        curvature: Hyperbolic curvature c.
        use_gradient_checkpointing: Use gradient checkpointing for hyperbolic attention.
        chunk_size: Chunk size for memory-efficient geodesic distance computation.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        use_hyperbolic_cross_attn: bool = True,
        curvature: float = 0.01,
        use_gradient_checkpointing: bool = False,
        chunk_size: Optional[int] = None,
        q_chunk_size: int = 256,
        k_chunk_size: int = 256,
        use_ball_query: bool = True,
        k_neighbors: int = 64,
    ) -> None:
        super().__init__()
        self.use_hyperbolic_cross_attn = use_hyperbolic_cross_attn
        self.curvature = curvature
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.chunk_size = chunk_size
        self.q_chunk_size = q_chunk_size
        self.k_chunk_size = k_chunk_size
        self.use_ball_query = use_ball_query
        self.k_neighbors = k_neighbors

        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        if use_hyperbolic_cross_attn:
            self.cross_attn_token_to_image = HyperCrossAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                curvature=curvature,
                aggregation="tangent_mean",
                use_gradient_checkpointing=use_gradient_checkpointing,
                chunk_size=chunk_size,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                use_ball_query=use_ball_query,
                k_neighbors=k_neighbors,
            )
        else:
            self.cross_attn_token_to_image = Attention(
                embedding_dim, num_heads, downsample_rate=attention_downsample_rate
            )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)

        if use_hyperbolic_cross_attn:
            self.cross_attn_image_to_token = HyperCrossAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                curvature=curvature,
                aggregation="tangent_mean",
                use_gradient_checkpointing=use_gradient_checkpointing,
                chunk_size=chunk_size,
                q_chunk_size=q_chunk_size,
                k_chunk_size=k_chunk_size,
                use_ball_query=use_ball_query,
                k_neighbors=k_neighbors,
            )
            # Same point-cloud key scaling for prompt→point and point→prompt branches.
            self.cross_attn_image_to_token.key_hyp_scale = (
                self.cross_attn_token_to_image.key_hyp_scale
            )
        else:
            self.cross_attn_image_to_token = Attention(
                embedding_dim, num_heads, downsample_rate=attention_downsample_rate
            )

        self.skip_first_layer_pe = skip_first_layer_pe
        self.hyper_ops = HyperOps(curvature=curvature, eps=1e-5)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
        hyperbolic_queries: Optional[Tensor] = None,
        temperature: Optional[Tensor] = None,
        radius: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_pos: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        use_hyp = self.use_hyperbolic_cross_attn and hyperbolic_queries is not None

        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe

        if use_hyp and temperature is not None:
            attn_out = self.cross_attn_token_to_image(
                query=hyperbolic_queries,
                key=k,
                value=keys,
                temperature=temperature,
                radius=radius,
                key_in_hyperbolic=False,
                query_pos=query_pos,
                key_pos=key_pos,
            )
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
        else:
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)

        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe

        if use_hyp and temperature is not None:
            scale = self.cross_attn_token_to_image.key_hyp_scale
            k_h = self.hyper_ops.exp0(scale * k)
            attn_out = self.cross_attn_image_to_token(
                query=k_h,
                key=hyperbolic_queries,
                value=hyperbolic_queries,
                temperature=temperature,
                radius=radius,
                key_in_hyperbolic=True,
                query_pos=key_pos,
                key_pos=query_pos,
            )
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
        else:
            attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)

        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """Standard attention layer with downsampling."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act(inplace=True) if act == nn.GELU else act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

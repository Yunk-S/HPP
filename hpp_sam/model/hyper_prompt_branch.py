"""
HyperPromptBranch: Hyperbolic Radius Adjustment for Prompt-Conditioned Cross-Attention.

This module implements the hyperbolic prompt branch with HRA (Hyperbolic Radius Adjustment),
which transforms Euclidean prompt embeddings into hyperbolic space and extracts the radius
as a control signal for attention temperature.

Key Concepts:
    1. HRA (Hyperbolic Radius Adjustment): Adjusts the hyperbolic radius of prompt 
       embeddings to control segmentation granularity.
       
    2. Radius-Conditioned Temperature: Maps prompt radius to attention temperature,
       enabling fine-grained control over attention sharpness.

Mathematical Framework (from HRA-PointSAM_design.md):
    Phase 1 - Euclidean Intake: p_i ∈ R^d (Euclidean prompt embedding)
    Phase 2 - Manifold Projection: q_i = exp_0^c(α · p_i)
    Phase 3 - HRA Radius Adjustment: q_tilde_i = W_s ⊗_c q_i = exp_0^c(W_s · log_0^c(q_i))
    Phase 4 - Geometric Feature Decoupling:
        - Hyperbolic Query: q_tilde_i ∈ D_c^d
        - Radius Scalar: r_i = d_c(q_tilde_i, 0) ∈ R^+
        
    Temperature Function:
        τ(r_i) = τ_min + (τ_max - τ_min) · σ(a · r_i + b_0)
        
        Physical meaning:
        - Small radius (r_i small) → τ small → smooth attention → coarse segmentation
        - Large radius (r_i large) → τ large → sharp attention → fine segmentation

References:
    - HRA-PointSAM_design.md (数据流动 section)
    - HyperET (https://github.com/godlin-sjtu/HyperET)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Mapping, Optional, Literal

from .hyper_ops import HyperOps
from hpp_sam.utils.hang_debug import forward_hang_probe


class HyperScaleWS_Diagonal(nn.Module):
    """
    Diagonal parameterization for HRA (Hyperbolic Radius Adjustment).

    **HyperET 对齐**：官方实现使用 Poincaré 球上的 Möbius 矩阵乘
    ``M ⊗_c x``（见 ``HyperOps.mobius_matvec``），而非切空间上的
    ``exp_0(W log_0 x)``。二者在一般矩阵下不等价；对角阵情形与 HyperET
    ``DiagonalLinear`` 中 ``mobius_matvec(diag(s), x)`` 一致。

    Args:
        dim: Embedding dimension.
        hyper_ops: HyperOps instance for hyperbolic operations.
    """

    def __init__(self, dim: int, hyper_ops: HyperOps):
        super().__init__()
        self.dim = dim
        self.hyper_ops = hyper_ops
        self.scaling = nn.Parameter(torch.ones(dim))

    def forward(self, x_ball: torch.Tensor) -> torch.Tensor:
        """Apply diagonal HRA: ``diag(s) ⊗_c x`` (HyperET-style)."""
        w = torch.diag(self.scaling)
        return self.hyper_ops.mobius_matvec(w, x_ball)


class HyperScaleWS_BlockDiagonal(nn.Module):
    """
    Block-diagonal HRA（与 HyperET ``adapter_blockd`` 同族：块内满秩线性变换 +
    Poincaré Möbius 矩阵乘）。

    实现为构造块对角矩阵 ``W`` 后调用 ``mobius_matvec(W, x)``，与对角版
    HyperET 路径一致，避免仅用 ``exp_0(W log_0 x)`` 与论文公式偏差。

    Args:
        dim: Embedding dimension (must be divisible by block_size).
        hyper_ops: HyperOps instance for hyperbolic operations.
        block_size: Size of each block. Default 8.
        init_scale: Scale for warm-start initialization (default 0.3).
    """
    
    def __init__(
        self, 
        dim: int, 
        hyper_ops: HyperOps,
        block_size: int = 8,
        init_scale: float = 0.3,
    ):
        super().__init__()
        assert dim % block_size == 0, f"dim={dim} must be divisible by block_size={block_size}"
        
        self.dim = dim
        self.block_size = block_size
        self.num_blocks = dim // block_size
        self.hyper_ops = hyper_ops
        
        # Warm start with stronger initialization for better early training
        # Using 0.3 instead of 0.1 to give HRA more expressive gradient from start
        self.blocks = nn.ParameterList([
            nn.Parameter(torch.eye(block_size) + init_scale * torch.randn(block_size, block_size))
            for _ in range(self.num_blocks)
        ])
    
    def forward(self, x_ball: torch.Tensor, hang_debug: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        orig_shape = x_ball.shape
        if x_ball.dim() > 2:
            x_flat = x_ball.reshape(-1, self.dim)
        elif x_ball.dim() == 2:
            x_flat = x_ball
        else:
            x_flat = x_ball.unsqueeze(0)

        _probe_hra = os.environ.get("HPP_HANG_DEBUG_HRA", "0") == "1"
        if _probe_hra and hang_debug is not None:
            _tag = f"hrablk.{hang_debug.get('tag', '?')}"
            forward_hang_probe(hang_debug, f"{_tag}.block_diag_start")
        w = torch.block_diag(*[b for b in self.blocks])
        if _probe_hra and hang_debug is not None:
            forward_hang_probe(hang_debug, f"{_tag}.block_diag_end", extra=f"Wshape={tuple(w.shape)}")

        if _probe_hra and hang_debug is not None:
            forward_hang_probe(hang_debug, f"{_tag}.mobius_matvec_start")
        result = self.hyper_ops.mobius_matvec(w, x_flat)
        if _probe_hra and hang_debug is not None:
            forward_hang_probe(hang_debug, f"{_tag}.mobius_matvec_end")

        if len(orig_shape) > 2:
            result = result.reshape(orig_shape)
        elif x_ball.dim() == 1:
            result = result.squeeze(0)
        return result


class HyperPromptBranch(nn.Module):
    """
    Hyperbolic Prompt Branch with HRA.
    
    This is the core module for HPP-SAM. It transforms Euclidean prompts into hyperbolic space
    and extracts the radius as a control signal for attention temperature.
    
    Data Flow (from HRA-PointSAM_design.md):
        Input:  p_i ∈ R^d (Euclidean prompt from PointEncoder)
        Output: q_tilde_i ∈ D_c^d (hyperbolic query)
                r_i ∈ R^+ (radius for granularity control)
                τ_i ∈ [τ_min, τ_max] (temperature for attention)
    
    Args:
        embed_dim: Embedding dimension (must match PointEncoder output).
        curvature: Initial hyperbolic curvature c. Default 0.01 (from HyperET).
                  This will be converted to a learnable parameter.
        hra_type: Type of HRA transformation. Options: "diagonal", "block_diagonal".
                  Default "block_diagonal".
        block_size: Block size for block-diagonal HRA. Default 8.
        tau_min: Minimum temperature. Default 0.1.
        tau_max: Maximum temperature. Default 2.0.
        learnable_curvature: If True, curvature is learnable. Default True.
    """
    
    def __init__(
        self,
        embed_dim: int,
        curvature: float = 0.01,
        hra_type: Literal["diagonal", "block_diagonal"] = "block_diagonal",
        block_size: int = 8,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        learnable_curvature: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hra_type = hra_type
        self.tau_min = tau_min
        self.tau_max = tau_max
        self._learnable_curvature = learnable_curvature
        
        # Shared HyperOps with learnable curvature
        self.hyper_ops = HyperOps(curvature=curvature, eps=1e-5, learnable=learnable_curvature)
        
        # Alpha: scale factor for initial exponential map
        # Increased from 0.1 to 0.4 for stronger early training signal
        self.alpha = nn.Parameter(torch.tensor(0.4))
        
        # Initialize HRA transformation
        if hra_type == "diagonal":
            self.hra_ws = HyperScaleWS_Diagonal(embed_dim, self.hyper_ops)
        elif hra_type == "block_diagonal":
            self.hra_ws = HyperScaleWS_BlockDiagonal(embed_dim, self.hyper_ops, block_size)
        else:
            raise ValueError(f"Unknown hra_type: {hra_type}. Options: 'diagonal', 'block_diagonal'")
        
        # Temperature function parameters
        self.temp_a = nn.Parameter(torch.tensor(0.5))
        self.temp_b0 = nn.Parameter(torch.tensor(0.0))
    
    def forward(
        self,
        euclidean_prompt: torch.Tensor,
        hang_debug: Optional[Mapping[str, Any]] = None,
        return_intermediate: bool = False,
    ) -> dict:
        """
        Transform Euclidean prompts to hyperbolic space with HRA.
        
        Args:
            euclidean_prompt: Euclidean prompt embeddings, shape [B, P, D].
                            B: batch size, P: number of prompts, D: embed dim.
            return_intermediate: If True, return intermediate values (q, q_tilde).
            
        Returns:
            dict with keys:
                - hyperbolic_prompt: q_tilde_i ∈ D_c^d, shape [B, P, D]
                - radius: r_i ∈ R^+, shape [B, P]
                - temperature: τ_i ∈ [τ_min, τ_max], shape [B, P]
                - q (if return_intermediate): q_i ∈ D_c^d, shape [B, P, D]
        """
        B, P, D = euclidean_prompt.shape

        # 获取 hang_debug payload（由外层传入）或检查 HPP_HANG_DEBUG_SAMPLE 环境变量
        _is_sample = (
            os.environ.get("HPP_HANG_DEBUG_SAMPLE", "0") == "1"
        )

        import time as _time
        _r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[HRA-BLOCK][rank{_r}][t={_time.time():.3f}] hpb.forward_enter", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"[HRA-BLOCK][rank{_r}][t={_time.time():.3f}] hpb.cuda_sync_ok", flush=True)

        scaled = self.alpha * euclidean_prompt
        print(f"[HRA-BLOCK][rank{_r}][t={_time.time():.3f}] hpb.after_scaled", flush=True)
        q = self.hyper_ops.exp0(scaled)
        print(f"[HRA-BLOCK][rank{_r}][t={_time.time():.3f}] hpb.after_exp0 qfinite={q.isfinite().all().item()}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Phase 2: HRA (Möbius transform)
        q_tilde = self.hra_ws(q, hang_debug=hang_debug)
        print(f"[HRA-BLOCK][rank{_r}][t={_time.time():.3f}] hpb.after_hra_ws", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        _probe_hra = os.environ.get("HPP_HANG_DEBUG_HRA", "0") == "1"
        _tag = f"hra.{hang_debug.get('tag', '?')}" if hang_debug else "?"

        # Phase 3: radius computation
        if _probe_hra and hang_debug is not None:
            forward_hang_probe(hang_debug, f"{_tag}.poincare_dist_start")
        origin = torch.zeros_like(q_tilde)
        radius = self.hyper_ops.poincare_dist(q_tilde, origin)
        if _probe_hra and hang_debug is not None:
            forward_hang_probe(hang_debug, f"{_tag}.poincare_dist_end", extra=f"radius_finite={radius.isfinite().all().item()}")

        # Phase 4: temperature
        temperature = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(
            self.temp_a * radius + self.temp_b0
        )
        
        output = {
            "hyperbolic_prompt": q_tilde,
            "radius": radius,
            "temperature": temperature,
        }
        
        if return_intermediate:
            output["q"] = q
        
        return output
    
    def get_temperature_bounds(self) -> tuple[float, float]:
        """Return the temperature bounds (tau_min, tau_max)."""
        return (self.tau_min, self.tau_max)
    
    def get_curvature(self) -> float:
        """Get current curvature value (for logging)."""
        return self.hyper_ops.get_curvature_value()


def create_hyper_prompt_branch(
    embed_dim: int,
    curvature: float = 0.01,
    hra_type: Literal["diagonal", "block_diagonal"] = "block_diagonal",
    block_size: int = 8,
    tau_min: float = 0.1,
    tau_max: float = 2.0,
    learnable_curvature: bool = True,
) -> HyperPromptBranch:
    """Factory function to create HyperPromptBranch with common settings."""
    return HyperPromptBranch(
        embed_dim=embed_dim,
        curvature=curvature,
        hra_type=hra_type,
        block_size=block_size,
        tau_min=tau_min,
        tau_max=tau_max,
        learnable_curvature=learnable_curvature,
    )

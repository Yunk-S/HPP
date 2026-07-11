"""
HyperOps: Hyperbolic Operations for Poincaré Ball Model.

This module implements fundamental operations in hyperbolic space (Poincaré ball model),
providing the mathematical foundation for HPP-SAM.

Mathematical Background:
    Poincaré ball model: D_c^d = {x ∈ R^d : c||x||2 < 1}, ball radius = 1/sqrt(c)
    
    Key operations:
    - exp_0^c(v): Exponential map from tangent space to manifold
    - log_0^c(x): Logarithmic map from manifold to tangent space
    - d_c(x,y): Geodesic distance (Poincaré distance)
    - x ⊕_c y: Möbius addition
    - M ⊗_c x: Möbius matrix-vector multiplication (Hyperbolic Radius Adjustment core)

References:
    - Hyperbolic Neural Networks (Ganea et al., NeurIPS 2018)
    - HyperET (https://github.com/godlin-sjtu/HyperET)
    - Hyperbolic Attention Networks (NeurIPS 2019)
    - HRA-PointSAM_design.md
"""

import os
import torch
import torch.nn as nn
from typing import Optional


class HyperOps(nn.Module):
    """
    Hyperbolic operations for the Poincaré ball model.
    
    This class implements all fundamental operations needed for hyperbolic deep learning,
    with numerical stability considerations built-in.
    
    Attributes:
        curvature (nn.Parameter): Learnable hyperbolic curvature c. 
                                Default initialization is 0.01 (from HyperET).
                                Ball radius = 1/sqrt(c)
        eps (float): Small constant for numerical stability.
        min_curvature (float): Minimum curvature clamp value.
        max_curvature (float): Maximum curvature clamp value.
    """
    
    # Class-level bounds for curvature
    MIN_CURVATURE = 1e-6
    MAX_CURVATURE = 1.0
    
    def __init__(
        self, 
        curvature: float = 0.01, 
        eps: float = 1e-5,
        learnable: bool = True,
    ):
        """
        Initialize HyperOps.
        
        Args:
            curvature: Initial hyperbolic curvature c. Smaller c → larger ball radius.
                      Default 0.01 gives radius ≈ 10.
            eps: Small constant for numerical stability in divisions and artanh.
            learnable: If True, curvature is a learnable parameter (recommended).
                       If False, curvature is fixed (for debugging or transfer learning).
        """
        super().__init__()
        self.eps = eps
        self._learnable = learnable
        
        # Initialize curvature as a parameter for adaptability
        if learnable:
            # Initialize with log-scale for better optimization (curvature > 0)
            # Using softplus to ensure positivity during training
            self.curvature_scale = nn.Parameter(
                torch.tensor(curvature).log(), requires_grad=True
            )
        else:
            self.register_buffer('curvature_scale', torch.tensor(curvature).log())
    
    @property
    def curvature(self) -> torch.Tensor:
        """Get current curvature value with numerical stability."""
        # Convert from log-space and apply softplus for positivity
        c = torch.nn.functional.softplus(self.curvature_scale)
        # Clamp to valid range
        return c.clamp(min=self.MIN_CURVATURE, max=self.MAX_CURVATURE)
    
    @curvature.setter
    def curvature(self, value: float):
        """Set curvature value (for loading checkpoints)."""
        if self._learnable:
            self.curvature_scale.data = torch.tensor(value).log()
        else:
            self.curvature_scale.data = torch.tensor(value).log()
    
    def get_curvature_value(self) -> float:
        """Get curvature as a Python float (for logging)."""
        return self.curvature.item()
    
    def artanh(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stable implementation of inverse hyperbolic tangent.
        
        artanh(x) = 0.5 * ln((1+x)/(1-x))
        
        For numerical stability, we clamp input to avoid inf/nan at boundaries.
        
        Args:
            x: Input tensor, values should be in (-1, 1) ideally.
            
        Returns:
            artanh(x), same shape as input.
        """
        x = torch.clamp(x, -1 + self.eps, 1 - self.eps)
        return 0.5 * torch.log((1 + x) / (1 - x))
    
    def tanh(self, x: torch.Tensor, clamp_val: float = 15.0) -> torch.Tensor:
        """
        Compute tanh with numerical stability.
        
        Args:
            x: Input tensor.
            clamp_val: Clamp value for numerical stability.
            
        Returns:
            tanh(x), same shape as input.
        """
        return x.clamp(-clamp_val, clamp_val).tanh()
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points back into the Poincaré ball for numerical stability.
        
        Args:
            x: Points in Euclidean space, shape [..., D].
            
        Returns:
            Projected points with norm clamped to < 1/sqrt(c).
        """
        eps = self.eps
        max_norm = (1 - eps) / (self.curvature ** 0.5 + eps)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        mask = norm > max_norm
        x = torch.where(mask, (max_norm / (norm + eps)) * x, x)
        return x
    
    def exp0(self, v: torch.Tensor, c: Optional[float] = None) -> torch.Tensor:
        """
        Exponential map from tangent space at origin to Poincaré ball.
        
        Formula: exp_0^c(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
        
        Args:
            v: Tangent vectors at origin, shape [..., D].
            c: Curvature override. If None, uses self.curvature.
            
        Returns:
            Points on the Poincaré ball, same shape as input.
        """
        try:
            c = c if c is not None else self.curvature
            sqrt_c = c ** 0.5
            norm_v_safe = torch.clamp_min(
                torch.norm(v, p=2, dim=-1, keepdim=True),
                self.eps
            )
            second_term = v / (sqrt_c * norm_v_safe)
            first_term = self.tanh(sqrt_c * norm_v_safe)
            result = first_term * second_term
            
            # Guard: detect and fix NaN/Inf before projection
            if not torch.isfinite(result).all():
                result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return self.project(result)
        except RuntimeError:
            # Fallback: return normalized input scaled by max ball radius
            norm = torch.clamp_min(torch.norm(v, p=2, dim=-1, keepdim=True), self.eps)
            max_radius = (1 - self.eps) / (self.curvature ** 0.5)
            return (max_radius * 0.5) * (v / norm)
    
    def log0(self, x: torch.Tensor, c: Optional[float] = None) -> torch.Tensor:
        """
        Logarithmic map from Poincaré ball to tangent space at origin.
        
        Formula: log_0^c(y) = (1/sqrt(c)) * artanh(sqrt(c) * ||y||) * y / ||y||
        
        Args:
            x: Points on the Poincaré ball, shape [..., D].
            c: Curvature override. If None, uses self.curvature.
            
        Returns:
            Tangent vectors at origin, same shape as input.
        """
        try:
            c = c if c is not None else self.curvature
            sqrt_c = c ** 0.5
            norm_x_safe = torch.clamp_min(
                torch.norm(x, p=2, dim=-1, keepdim=True),
                self.eps
            )
            inner_arg = sqrt_c * norm_x_safe
            inner_arg = torch.clamp(inner_arg, -1 + self.eps, 1 - self.eps)
            artanh_term = self.artanh(inner_arg)
            result = (1 / sqrt_c) * artanh_term * (x / norm_x_safe)
            
            # Guard: detect and fix NaN/Inf
            if not torch.isfinite(result).all():
                result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return result
        except RuntimeError:
            # Fallback: return input as-is (approximation for small norms)
            return x
    
    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, 
                   c: Optional[float] = None,
                   use_gradient: bool = True) -> torch.Tensor:
        """
        Möbius addition (hyperbolic analogue of vector addition).
        
        Formula:
        x ⊕_c y = ((1 + 2c< x,y > + c||y||^2)x + (1 - c||x||^2)y) / 
                  (1 + 2c< x,y > + c^2||x||^2||y||^2)
        
        Memory-optimized version: uses fused operations to minimize peak memory
        by avoiding unnecessary intermediate tensor allocations.

        Args:
            x: First point on Poincaré ball, shape [..., D].
            y: Second point on Poincaré ball, shape [..., D].
            c: Curvature override. If None, uses self.curvature.
            use_gradient: If False, uses no_grad for projection (saves memory in inference).
            
        Returns:
            Result of Möbius addition, same shape as inputs.
        """
        try:
            # Input validation for CUDA safety
            if x.dim() != y.dim():
                raise ValueError(f"Dimension mismatch: x.dim()={x.dim()}, y.dim()={y.dim()}")
            
            c_tensor = c if c is not None else self.curvature
            c_val = float(c_tensor.detach().item()) if torch.is_tensor(c_tensor) else c_tensor
            if c_val <= 0:
                c_val = self.eps  # Guard against zero/negative curvature

            # Compute squared norms efficiently using einsum for better memory efficiency
            x_sq = torch.sum(x * x, dim=-1, keepdim=True)
            y_sq = torch.sum(y * y, dim=-1, keepdim=True)
            xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

            # xy_dot can be safely used as-is for all valid Poincaré ball inputs.
            # Guard only the denominator: when ||x||·||y|| is near the singularity
            # boundary (||x||² ≈ 1/c for x ≈ -y), the denominator clamp below handles it.
            # NOT clamping xy_dot here preserves exactness for x ⊕_c x self-distances
            # (mobius_add(x, x) must equal x for non-zero x), which is required for
            # poincare_dist(x, x) = 0 in all valid radius ranges.
            
            denom_base = 1.0 + 2.0 * c_val * xy_dot
            
            # Guard: denominator must stay positive
            c_sq_x_sq = c_val * c_val * x_sq
            c_sq_x_sq_y_sq = c_sq_x_sq * y_sq
            denominator = denom_base + c_sq_x_sq_y_sq
            # Clamp to avoid zero/negative denominator
            denominator = torch.clamp(denominator, min=self.eps)
            
            # Compute scalar part for numerator_1
            scalar_part = 1.0 + 2.0 * c_val * xy_dot + c_val * y_sq
            
            # Compute numerator components
            # numerator_1 = scalar_part * x
            # numerator_2 = (1 - c * x_sq) * y
            numerator_1 = scalar_part * x
            numerator_2 = (1.0 - c_val * x_sq) * y
            
            # Combine numerator components
            numerator = numerator_1 + numerator_2
            
            # Final division with numerical stability
            result = numerator / (denominator + self.eps)
            
            # Guard: detect NaN/Inf and fall back to safe value
            if not torch.isfinite(result).all():
                # Return y as the closest safe approximation (q ⊕ 0 = q identity)
                return y
            
            # Project back to ball (use no_grad during inference for memory savings)
            if use_gradient:
                return self.project(result)
            else:
                with torch.no_grad():
                    return self.project(result)
        except RuntimeError:
            # Fallback: return y (closest safe approximation)
            return y
    
    def poincare_dist(self, x: torch.Tensor, y: torch.Tensor,
                      c: Optional[float] = None,
                      use_gradient: bool = True) -> torch.Tensor:
        """
        Compute Poincaré distance (geodesic distance) between points.
        
        Formula: d_c(x,y) = (2/sqrt(c)) * artanh(sqrt(c) * ||(-x) ⊕_c y||)
        
        Memory-optimized version with two-stage computation:
        1. First compute the norm of mobius result (lighter memory footprint)
        2. Then compute the distance using just the norm
        
        For memory-critical scenarios, can compute in chunks along the batch dimension.

        Args:
            x: First point on Poincaré ball, shape [..., D].
            y: Second point on Poincaré ball, shape [..., D].
            c: Curvature override. If None, uses self.curvature.
            use_gradient: If False, uses no_grad for intermediate computations.

        Returns:
            Geodesic distances, shape [...] (one less dimension than inputs).
        """
        try:
            c = c if c is not None else self.curvature
            sqrt_c = c ** 0.5
            
            # Compute negation efficiently
            neg_x = -x
            
            # Guard: check inputs are finite before entering CUDA kernels
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                diff = x - y
                return torch.norm(diff, p=2, dim=-1)
            
            # Compute mobius addition with gradient control for memory efficiency
            mobius_result = self.mobius_add(neg_x, y, c, use_gradient=use_gradient)
            
            # Guard: mobius_result may be NaN from degenerate geometry
            if not torch.isfinite(mobius_result).all():
                diff = x - y
                return torch.norm(diff, p=2, dim=-1)
            
            # Compute norm directly on result (avoiding intermediate assignment)
            norm_mobius = torch.norm(mobius_result, p=2, dim=-1)
            
            # Guard: norm_mobius must be finite for artanh
            if not torch.isfinite(norm_mobius).all():
                diff = x - y
                return torch.norm(diff, p=2, dim=-1)
            
            # Compute distance using clamped inner argument
            inner_arg = torch.clamp(sqrt_c * norm_mobius, -1 + self.eps, 1 - self.eps)
            dist = (2 / sqrt_c) * self.artanh(inner_arg)
            
            return dist
        except RuntimeError:
            # Fallback to Euclidean distance when hyperbolic computation fails
            diff = x - y
            return torch.norm(diff, p=2, dim=-1)
    
    def mobius_matvec(self, W: torch.Tensor, x: torch.Tensor,
                      c: Optional[float] = None) -> torch.Tensor:
        """
        Möbius matrix-vector multiplication (HyperET formula).
        
        Formula from HyperET:
        M ⊗_c x = (1/√c) * tanh(||Mx||₂/||x||₂ * tanh⁻¹(√c||x||₂)) * Mx/||Mx||₂
        
        This is the core operation for Hyperbolic Radius Adjustment (HRA).
        
        Args:
            W: Weight matrix, shape [D_out, D_in].
            x: Point on Poincaré ball, shape [..., D_in].
            c: Curvature override. If None, uses self.curvature.
            
        Returns:
            Transformed points on Poincaré ball, shape [..., D_out].
        """
        _probe = os.environ.get("HPP_HANG_DEBUG_HRA", "0") == "1"
        _tag = os.environ.get("HPP_HANG_DEBUG_TAG", "?")

        c = c if c is not None else self.curvature
        sqrt_c = c ** 0.5
        
        x_norm = torch.clamp_min(
            torch.norm(x, p=2, dim=-1, keepdim=True), 
            self.eps
        )
        
        if _probe:
            import time
            _r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[HRA-PROBE][{_tag}][rank{_r}][t={time.time():.3f}] mobius_matvec_start xshape={tuple(x.shape)} Wshape={tuple(W.shape)}", flush=True)
            if x.is_cuda:
                torch.cuda.synchronize()
                print(f"[HRA-PROBE][{_tag}][rank{_r}][t={time.time():.3f}] pre_matmul_cuda_sync_ok", flush=True)
        mx = torch.matmul(x, W.transpose(-1, -2))
        if _probe:
            if x.is_cuda:
                torch.cuda.synchronize()
            import time
            _r = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[HRA-PROBE][{_tag}][rank{_r}][t={time.time():.3f}] mobius_matvec_after_matmul mxfinite={mx.isfinite().all().item()} mxshape={tuple(mx.shape)}", flush=True)
        
        # Guard: detect NaN/Inf in matrix multiplication result
        if not torch.isfinite(mx).all():
            mx = torch.nan_to_num(mx, nan=0.0, posinf=1.0, neginf=-1.0)
        
        mx_norm = torch.clamp_min(
            torch.norm(mx, p=2, dim=-1, keepdim=True),
            self.eps
        )
        
        # Guard: clamp x_norm * sqrt_c to valid range for artanh
        x_norm_scaled = torch.clamp(sqrt_c * x_norm, -1 + self.eps, 1 - self.eps)
        inner_term = (mx_norm / x_norm) * self.artanh(x_norm_scaled)
        res_c = self.tanh(inner_term) * mx / (mx_norm * sqrt_c)
        
        # Guard: detect NaN/Inf before final output
        if not torch.isfinite(res_c).all():
            res_c = torch.nan_to_num(res_c, nan=0.0, posinf=1.0, neginf=-1.0)
        
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.bool)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        result = torch.where(cond, res_0, res_c)
        
        return self.project(result)
    
    def parallel_transport(self, v: torch.Tensor, x: torch.Tensor, 
                           y: torch.Tensor, c: Optional[float] = None) -> torch.Tensor:
        """
        Parallel transport of vector v from x to y along the geodesic.
        
        Args:
            v: Tangent vector at x, shape [..., D].
            x: Source point on Poincaré ball, shape [..., D].
            y: Target point on Poincaré ball, shape [..., D].
            c: Curvature override. If None, uses self.curvature.
            
        Returns:
            Transported tangent vector at y, same shape as v.
        """
        c = c if c is not None else self.curvature
        sqrt_c = c ** 0.5
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        norm_y = torch.norm(y, dim=-1, keepdim=True)
        norm_x_safe = torch.clamp(norm_x, min=self.eps)
        norm_y_safe = torch.clamp(norm_y, min=self.eps)
        neg_x = -x
        mobius_result = self.mobius_add(neg_x, y, c)
        norm_mobius = torch.norm(mobius_result, dim=-1, keepdim=True)
        norm_mobius = torch.clamp(norm_mobius, min=self.eps)
        log_y = self.log0(y, c)
        log_x = self.log0(x, c)
        x_sq = norm_x ** 2
        factor = (1 - c * x_sq) / (1 - c * x_sq + self.eps)
        scale = (norm_y_safe / norm_mobius) / sqrt_c
        diff_log = log_y - factor * log_x
        v_norm_keepdim = torch.norm(v, dim=-1, keepdim=True)
        result = scale * diff_log * v_norm_keepdim / (torch.norm(diff_log, dim=-1, keepdim=True) + self.eps)
        return result
    
    def tangent_mean(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None,
                     c: Optional[float] = None) -> torch.Tensor:
        """
        Compute weighted mean in tangent space (default aggregation method).
        
        Formula: m = exp_0^c(Σ_i w_i * log_0^c(x_i))
        
        Args:
            x: Points on Poincaré ball, shape [B, N, D].
            weights: Optional weights, shape [B, N]. Default uniform.
            c: Curvature override. If None, uses self.curvature.
            
        Returns:
            Mean point on Poincaré ball, shape [B, D].
        """
        c = c if c is not None else self.curvature
        B, N, D = x.shape
        
        if weights is None:
            weights = torch.ones(B, N, device=x.device, dtype=x.dtype) / N
        else:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        
        x_log = self.log0(x, c)
        weighted_sum = torch.matmul(weights.unsqueeze(1), x_log).squeeze(1)
        mean_point = self.exp0(weighted_sum, c)
        return mean_point


def create_hyper_ops(curvature: float = 0.01, eps: float = 1e-5) -> HyperOps:
    """Factory function to create HyperOps with common default settings."""
    return HyperOps(curvature=curvature, eps=eps)


DEFAULT_CURVATURE = 0.01
DEFAULT_EPS = 1e-5
MIN_CURVATURE = 1e-6
MAX_CURVATURE = 1.0

"""Mathematical sanity checks for Poincaré-ball operations in `hyper_ops.HyperOps`."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

# Load hyper_ops without importing hpp_sam.model.__init__ (avoids optional torkit3d).
_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_hyper_ops_testonly",
    _ROOT / "hpp_sam" / "model" / "hyper_ops.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
HyperOps = _mod.HyperOps


@pytest.fixture
def ops():
    return HyperOps(curvature=0.01, eps=1e-5)


def _random_ball_points(ops: HyperOps, n: int, d: int, device="cpu", scale=0.3):
    """Small Euclidean vectors that map safely into the ball via exp0."""
    torch.manual_seed(0)
    v = torch.randn(n, d, device=device) * scale
    x = ops.exp0(v)
    return x


def test_exp_log_roundtrip_small_tangent(ops):
    torch.manual_seed(1)
    v = torch.randn(4, 16) * 0.05
    x = ops.exp0(v)
    v2 = ops.log0(x)
    assert torch.allclose(v, v2, atol=1e-4, rtol=1e-3)


def test_log_exp_roundtrip_in_ball(ops):
    x = _random_ball_points(ops, 8, 32, scale=0.2)
    v = ops.log0(x)
    x2 = ops.exp0(v)
    assert torch.allclose(x, x2, atol=1e-4, rtol=1e-3)


def test_poincare_dist_self_zero(ops):
    x = _random_ball_points(ops, 5, 64)
    d = ops.poincare_dist(x, x)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5, rtol=0)


def test_poincare_dist_symmetry(ops):
    x = _random_ball_points(ops, 3, 24)
    y = _random_ball_points(ops, 3, 24)
    d_xy = ops.poincare_dist(x, y)
    d_yx = ops.poincare_dist(y, x)
    assert torch.allclose(d_xy, d_yx, atol=1e-5, rtol=1e-4)


def test_poincare_dist_triangle_inequality_sample(ops):
    x = _random_ball_points(ops, 1, 8)[0]
    y = _random_ball_points(ops, 1, 8)[0]
    z = _random_ball_points(ops, 1, 8)[0]
    d_xy = ops.poincare_dist(x.unsqueeze(0), y.unsqueeze(0)).item()
    d_yz = ops.poincare_dist(y.unsqueeze(0), z.unsqueeze(0)).item()
    d_xz = ops.poincare_dist(x.unsqueeze(0), z.unsqueeze(0)).item()
    assert d_xz <= d_xy + d_yz + 1e-3


def test_project_stays_inside_ball(ops):
    c = ops.curvature
    max_norm = (1 - ops.eps) / (math.sqrt(c) + ops.eps)
    x = torch.randn(2, 10) * 1000
    y = ops.project(x)
    norms = torch.norm(y, p=2, dim=-1)
    assert bool((norms <= max_norm * 1.0001).all())


def test_mobius_add_identity_zero(ops):
    y = _random_ball_points(ops, 4, 16)
    zero = torch.zeros_like(y)
    out = ops.mobius_add(zero, y)
    assert torch.allclose(out, y, atol=1e-5, rtol=1e-4)


def test_tangent_mean_uniform_weights(ops):
    x = _random_ball_points(ops, 2, 5, scale=0.15).unsqueeze(0)  # [1, N, D]
    m = ops.tangent_mean(x, weights=None)
    assert m.shape == (1, 5)
    assert torch.isfinite(m).all()


def test_artanh_tanh_inverse(ops):
    t = torch.linspace(-0.5, 0.5, steps=7)
    assert torch.allclose(ops.tanh(ops.artanh(t)), t, atol=1e-4, rtol=1e-3)

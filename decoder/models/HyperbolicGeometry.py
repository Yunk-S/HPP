"""FP32 Poincare-ball entailment geometry (Ganea et al., ICML 2018).

rho denotes Euclidean ball radius, in units of 1/sqrt(c). All geometry
outputs stay FP32, even inside autocast; casting happens only at the gate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class HyperbolicGeometry(nn.Module):
    def __init__(self, c=1.0, eps=1e-6, cone_k=0.1):
        super().__init__()
        if c <= 0 or not 0 < eps < 0.01 or cone_k <= 0:
            raise ValueError('c, cone_k must be positive; eps must be in (0, .01)')
        self.c, self.sqrt_c, self.eps, self.cone_k = c, math.sqrt(c), eps, cone_k

    def project(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            return x * ((1 - self.eps) / (self.sqrt_c * norm)).clamp(max=1)

    def expmap0(self, v):
        with torch.autocast(device_type=v.device.type, enabled=False):
            v = v.float()
            norm = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            return self.project(torch.tanh(self.sqrt_c * norm) * v / (self.sqrt_c * norm))

    def construct_query(self, direction, radius):
        with torch.autocast(device_type=direction.device.type, enabled=False):
            d = direction.float()
            # A deterministic direction avoids zero vectors at the cone origin.
            norm = d.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(d)
            fallback[..., 0] = 1
            d = torch.where(norm > self.eps, d / norm.clamp_min(self.eps), fallback)
            return self.project(d * radius.float().unsqueeze(-1))

    def aperture(self, q):
        with torch.autocast(device_type=q.device.type, enabled=False):
            r = (self.sqrt_c * self.project(q)).norm(dim=-1).clamp_min(self.eps)
            return torch.asin((self.cone_k * (1 - r.square()) / r).clamp(0, 1 - self.eps))

    def cone_angle_closed_form(self, q, k):
        """q [B,D], k [B,N,D] -> Xi [B,N]. Coincident points entail themselves."""
        with torch.autocast(device_type=q.device.type, enabled=False):
            x, y = self.sqrt_c * self.project(q), self.sqrt_c * self.project(k)
            x2 = x.square().sum(-1).unsqueeze(-1)
            y2 = y.square().sum(-1)
            xy = (x.unsqueeze(1) * y).sum(-1)
            diff = (x.unsqueeze(1) - y).norm(dim=-1)
            numerator = xy * (1 + x2) - x2 * (1 + y2)
            denominator = x2.clamp_min(self.eps**2).sqrt() * diff.clamp_min(self.eps) * (
                1 + x2 * y2 - 2 * xy).clamp_min(self.eps**2).sqrt()
            cosine = (numerator / denominator.clamp_min(self.eps**2)).clamp(-1 + self.eps, 1 - self.eps)
            return torch.where(diff <= self.eps, torch.zeros_like(cosine), torch.acos(cosine))


class RadiusController(nn.Module):
    def __init__(self, rho_min=0.15, rho_max=0.85, c=1.0, cone_k=0.1):
        super().__init__()
        lower = 2 * cone_k / (math.sqrt(1 + 4 * cone_k**2) + 1)
        if not lower < math.sqrt(c) * rho_min < math.sqrt(c) * rho_max < 1 - 1e-5:
            raise ValueError('radii must be ordered and inside the valid cone annulus')
        self.rho_min, self.rho_max = rho_min, rho_max
        self.raw_gamma = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    @property
    def gamma(self):
        return F.softplus(self.raw_gamma.float()).clamp_min(1e-4)

    def forward(self, g):
        with torch.autocast(device_type=g.device.type, enabled=False):
            g = g.float()
            if not torch.isfinite(g).all() or ((g < 0) | (g > 1)).any():
                raise ValueError('granularity must be finite and in [0,1]')
            # Preserve exact endpoints without 0**gamma's singular derivative.
            power = torch.exp(self.gamma * g.clamp_min(1e-12).log())
            power = torch.where(g == 0, torch.zeros_like(power), power)
            return self.rho_min + (self.rho_max - self.rho_min) * power


class EntailmentConeGate(nn.Module):
    def __init__(self, feature_dim=384, hyper_dim=32, c=1.0,
                 rho_min=0.15, rho_max=0.85, key_radius=0.95, cone_k=0.1, beta=5.0):
        super().__init__()
        if not rho_max < key_radius < (1 - 1e-5) / math.sqrt(c) or beta <= 0:
            raise ValueError('key_radius must exceed rho_max and remain inside the ball; beta > 0')
        self.semantic_projection = nn.Linear(feature_dim, hyper_dim)
        self.geometry = HyperbolicGeometry(c=c, cone_k=cone_k)
        self.radius_controller = RadiusController(rho_min, rho_max, c, cone_k)
        self.key_radius = key_radius
        self.raw_beta = nn.Parameter(torch.tensor(math.log(math.expm1(beta))))

    def forward(self, features, prompt_indices, g):
        with torch.autocast(device_type=features.device.type, enabled=False):
            h = F.linear(features.float(), self.semantic_projection.weight.float(),
                         self.semantic_projection.bias.float())
            rho = self.radius_controller(g)
            prompt = h[torch.arange(h.shape[0], device=h.device), prompt_indices]
            q = self.geometry.construct_query(prompt, rho)
            k = self.geometry.construct_query(h, torch.full(h.shape[:2], self.key_radius, device=h.device))
            angles = self.geometry.cone_angle_closed_form(q, k)
            psi = self.geometry.aperture(q)
            energy = F.relu(angles - psi.unsqueeze(-1))
            beta = F.softplus(self.raw_beta.float())
            return {'gates': torch.exp(-beta * energy), 'energy': energy,
                    'rho': rho, 'psi': psi, 'angles': angles}

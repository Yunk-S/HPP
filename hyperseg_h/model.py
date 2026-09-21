import torch
from torch import nn
from decoder.models.PointFeatureEnhancer import PointFeatureEnhancer
from decoder.models.CrossAttentionDecoder import Decoder
from decoder.models.seghead_mlp import SegHead
from decoder.models.HyperbolicGeometry import EntailmentConeGate


class HyperSegH(nn.Module):
    """Cached official PVCNN features -> prompt-conditioned segmentation.

    control_signal describes the caller's value, never inferred from its magnitude.
    An optional backbone accepts normalized [B,N,3] and returns [B,N,feature_dim].
    """
    def __init__(self, feature_dim=448, hidden_dim=384, hyper_dim=32, num_heads=8,
                 enhancer_layers=2, decoder_layers=4, dropout=0.1,
                 compatibility_mode='official-compatibility', model_variant='hyperseg-h',
                 control_signal='scale-proxy', hierarchy_enabled=False,
                 backbone=None, freeze_backbone=True, **geometry):
        super().__init__()
        if hidden_dim % 6 or hidden_dim % num_heads:
            raise ValueError('hidden_dim must be divisible by 6 and num_heads')
        if model_variant not in ('legacy', 'hyperseg-h', 'radius-film', 'spherical-hierarchy'):
            raise ValueError('Unknown model_variant')
        if control_signal not in ('scale', 'scale-proxy', 'hierarchy'):
            raise ValueError('Unknown control_signal')
        if (control_signal == 'hierarchy') != hierarchy_enabled:
            raise ValueError('hierarchy_enabled must explicitly match control_signal=hierarchy')
        if model_variant != 'legacy' and control_signal == 'scale':
            raise ValueError('Use scale-proxy for HyperSeg-H A1; scale is reserved for legacy')
        self.control_signal, self.model_variant = control_signal, model_variant
        self.backbone, self.freeze_backbone = backbone, freeze_backbone
        if backbone is not None and freeze_backbone:
            backbone.requires_grad_(False)
        self.enhancer = PointFeatureEnhancer(
            feature_dim=feature_dim, feature_proj_dim=hidden_dim, pos_num_feats=hidden_dim // 3,
            transformer_hidden_dim=hidden_dim, transformer_num_heads=num_heads,
            transformer_num_layers=enhancer_layers, dropout=dropout,
            use_continuous_scale=model_variant in ('legacy', 'radius-film'),
            compatibility_mode=compatibility_mode)
        self.decoder = Decoder(hidden_dim, num_heads, decoder_layers, attn_drop=dropout,
                               proj_drop=dropout, drop_path=dropout,
                               gate_aware=model_variant in ('hyperseg-h', 'spherical-hierarchy'))
        self.seg_head = SegHead(hidden_dim, dropout)
        self.gate = EntailmentConeGate(hidden_dim, hyper_dim, **geometry) if model_variant != 'legacy' else None

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None and self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, points, prompt_indices, control, features=None, return_aux=False):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError('points must be [B,N,3]')
        b, n, _ = points.shape
        if prompt_indices.shape != (b,) or ((prompt_indices < 0) | (prompt_indices >= n)).any():
            raise ValueError('prompt_indices must be local indices [B] in [0,N)')
        if control.shape != (b,) or not torch.isfinite(control).all() or ((control < 0) | (control > 1)).any():
            raise ValueError('control must be [B] with finite values in [0,1]')
        if features is None:
            if self.backbone is None:
                raise ValueError('Supply cached PVCNN features or an explicit backbone')
            features = self.backbone(points)
        if features.shape[:2] != (b, n):
            raise ValueError('feature/point alignment mismatch')
        g = 1 - control if self.control_signal == 'scale-proxy' else control
        conditioning = control if self.model_variant == 'legacy' else None
        if self.model_variant == 'radius-film':
            conditioning = self.gate.radius_controller(g)
        enhanced = self.enhancer(features, points, continuous_scales=conditioning)
        prompt = enhanced[torch.arange(b, device=points.device), prompt_indices].unsqueeze(1)
        aux = {}
        gates = None
        if self.gate is not None and self.model_variant != 'radius-film':
            aux = self.gate(enhanced, prompt_indices, g)
            if self.model_variant == 'spherical-hierarchy':
                # Matched angular control: same projection, radius controller and aperture.
                with torch.autocast(device_type=points.device.type, enabled=False):
                    h = self.gate.semantic_projection(enhanced.float())
                    h = torch.nn.functional.normalize(h, dim=-1, eps=1e-6)
                    q = h[torch.arange(b, device=h.device), prompt_indices]
                    angle = torch.acos((h * q[:, None]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
                    aux['energy'] = torch.relu(angle - aux['psi'][:, None])
                    aux['gates'] = torch.exp(-torch.nn.functional.softplus(self.gate.raw_beta) * aux['energy'])
            gates = aux['gates']
        probs = self.seg_head(self.decoder(enhanced, prompt, gates=gates))
        return (probs, aux) if return_aux else probs

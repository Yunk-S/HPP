import torch
from torch import nn
from decoder.models.PointFeatureEnhancer import PointFeatureEnhancer
from decoder.models.CrossAttentionDecoder import Decoder
from decoder.models.seghead_mlp import SegHead
from decoder.models.HyperbolicGeometry import EntailmentConeGate


class HyperSegH(nn.Module):
    """Cached or raw-point PVCNN features -> prompt-conditioned segmentation.

    control_signal describes the caller's value, never inferred from its magnitude.
    An optional backbone accepts separately normalized [B,N,3] and returns
    [B,N,feature_dim].
    """
    def __init__(self, feature_dim=448, hidden_dim=384, hyper_dim=32, num_heads=8,
                 enhancer_layers=2, decoder_layers=4, dropout=0.1,
                 compatibility_mode='official-compatibility', model_variant='hyperseg-h',
                 control_signal='scale-proxy', hierarchy_enabled=False,
                 backbone=None, freeze_backbone=True, prompt_propagation=False, **geometry):
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
                               gate_aware=model_variant in ('hyperseg-h', 'spherical-hierarchy'),
                               prompt_propagation=prompt_propagation)
        self.seg_head = SegHead(hidden_dim, dropout)
        self.gate = EntailmentConeGate(hidden_dim, hyper_dim, **geometry) if model_variant != 'legacy' else None
        if model_variant == 'radius-film':
            # This ablation uses only the radius controller. Keep checkpoint
            # keys, but exclude inactive parameters from DDP's gradient reducer.
            self.gate.semantic_projection.requires_grad_(False)
            self.gate.raw_beta.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None and self.freeze_backbone:
            self.backbone.eval()
        return self

    @property
    def encoder_is_control_invariant(self):
        """Whether enhancer output can be reused for multiple controls."""
        return self.model_variant in ('hyperseg-h', 'spherical-hierarchy')

    def _validate_points(self, points, prompt_indices=None):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError('points must be [B,N,3]')
        b, n, _ = points.shape
        if prompt_indices is not None and (prompt_indices.shape != (b,) or
                                           ((prompt_indices < 0) | (prompt_indices >= n)).any()):
            raise ValueError('prompt_indices must be local indices [B] in [0,N)')
        return b, n

    def encode_features(self, points, features=None, control=None, backbone_points=None):
        """Run the optional backbone and enhancer once for a point cloud.

        HyperSeg-H's enhancer is independent of the hierarchy/control value,
        allowing training and evaluation to reuse this tensor across levels.
        ``backbone_points`` keeps raw-point encoder normalization separate from
        decoder positional coordinates.
        """
        b, n = self._validate_points(points)
        if features is None:
            if self.backbone is None:
                raise ValueError('Supply cached PVCNN features or an explicit backbone')
            source_points = points if backbone_points is None else backbone_points
            if source_points.shape != points.shape:
                raise ValueError('backbone_points must align with points')
            features = self.backbone(source_points)
        if features.shape[:2] != (b, n):
            raise ValueError('feature/point alignment mismatch')
        if control is None and not self.encoder_is_control_invariant:
            raise ValueError('control is required for a control-conditioned enhancer')
        conditioning = control if self.model_variant == 'legacy' else None
        if self.model_variant == 'radius-film' and control is not None:
            g = 1 - control if self.control_signal == 'scale-proxy' else control
            conditioning = self.gate.radius_controller(g)
        return self.enhancer(features, points, continuous_scales=conditioning)

    def decode_features(self, enhanced, prompt_indices, control, return_aux=False):
        if enhanced.ndim != 3:
            raise ValueError('enhanced features must be [B,N,C]')
        b, n, _ = enhanced.shape
        if prompt_indices.shape != (b,) or ((prompt_indices < 0) | (prompt_indices >= n)).any():
            raise ValueError('prompt_indices must be local indices [B] in [0,N)')
        if control.shape != (b,) or not torch.isfinite(control).all() or ((control < 0) | (control > 1)).any():
            raise ValueError('control must be [B] with finite values in [0,1]')
        g = 1 - control if self.control_signal == 'scale-proxy' else control
        prompt = enhanced[torch.arange(b, device=enhanced.device), prompt_indices].unsqueeze(1)
        aux = {}
        gates = None
        if self.gate is not None and self.model_variant != 'radius-film':
            aux = self.gate(enhanced, prompt_indices, g)
            if self.model_variant == 'spherical-hierarchy':
                # Matched angular control: same projection, radius controller and aperture.
                with torch.autocast(device_type=enhanced.device.type, enabled=False):
                    h = self.gate.semantic_projection(enhanced.float())
                    h = torch.nn.functional.normalize(h, dim=-1, eps=1e-6)
                    q = h[torch.arange(b, device=h.device), prompt_indices]
                    angle = torch.acos((h * q[:, None]).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))
                    aux['energy'] = torch.relu(angle - aux['psi'][:, None])
                    aux['gates'] = torch.exp(-torch.nn.functional.softplus(self.gate.raw_beta) * aux['energy'])
            gates = aux['gates']
        probs = self.seg_head(self.decoder(enhanced, prompt, gates=gates))
        return (probs, aux) if return_aux else probs

    def forward_hierarchy(self, points, prompt_indices, controls, features=None,
                          backbone_points=None, compute_midpoints=False):
        """Encode once and decode [B,L] controls within a single DDP forward.

        Control-dependent legacy/FiLM ablations reuse the backbone but must
        recompute the enhancer for each control, matching single-device behavior.
        """
        b, _ = self._validate_points(points, prompt_indices)
        if controls.ndim != 2 or controls.shape[0] != b or controls.shape[1] < 1:
            raise ValueError('hierarchy controls must be nonempty [B,L]')
        if features is None and self.backbone is not None:
            source = points if backbone_points is None else backbone_points
            if source.shape != points.shape:
                raise ValueError('backbone_points must align with points')
            features = self.backbone(source)
        elif features is not None and self.backbone is not None and self.training and not self.freeze_backbone:
            raise ValueError('A trainable backbone requires raw points, without cached features')
        enhanced = (self.encode_features(points, features, backbone_points=backbone_points)
                    if self.encoder_is_control_invariant else None)

        def decode(control):
            encoded = enhanced if enhanced is not None else self.encode_features(
                points, features, control, backbone_points)
            return self.decode_features(encoded, prompt_indices, control, return_aux=True)

        predictions, energies, middle = [], [], []
        for level in range(controls.shape[1]):
            pred, aux = decode(controls[:, level])
            predictions.append(pred)
            if 'energy' in aux:
                energies.append(aux['energy'])
        if compute_midpoints:
            for level in range(controls.shape[1] - 1):
                pred, _ = decode((controls[:, level] + controls[:, level + 1]) / 2)
                middle.append(pred)
        return {'predictions': torch.stack(predictions, 1),
                'energies': torch.stack(energies, 1) if energies else None,
                'middle': torch.stack(middle, 1) if middle else None}

    def forward(self, points, prompt_indices, control=None, features=None, return_aux=False,
                backbone_points=None, *, controls=None, compute_midpoints=False):
        if controls is not None:
            if control is not None:
                raise ValueError('Supply control or hierarchy controls, not both')
            return self.forward_hierarchy(points, prompt_indices, controls, features,
                                          backbone_points, compute_midpoints)
        self._validate_points(points, prompt_indices)
        enhanced = self.encode_features(points, features, control, backbone_points)
        return self.decode_features(enhanced, prompt_indices, control, return_aux)

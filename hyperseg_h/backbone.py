"""Lazy adapter around the official PVCNN + triplane feature extractor."""
import sys
from pathlib import Path
import torch
from torch import nn
from .checkpoint import audit_load


class OfficialBackbone(nn.Module):
    def __init__(self, config_path, checkpoint_path=None, load_checkpoint=True):
        super().__init__()
        encoder_root = str(Path(__file__).resolve().parents[1] / 'encoder')
        if encoder_root not in sys.path:
            sys.path.insert(0, encoder_root)
        from partfield.config.defaults import _C
        from partfield.model_trainer_pvcnn_only_demo import Model
        from partfield.model.PVCNN.encoder_pc import sample_triplane_feat
        cfg = _C.clone()
        cfg.merge_from_file(str(config_path))
        self.model = Model(cfg)
        self.audit = None
        if load_checkpoint:
            if not checkpoint_path:
                raise ValueError('An encoder checkpoint is required for backbone initialization')
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            state = state.get('state_dict', state)
            state = {k.removeprefix('module.').removeprefix('model.'): v for k, v in state.items()}
            self.audit = audit_load(self.model, state, 'official_encoder')
        self.sample = sample_triplane_feat
        # Reconstruction heads/logit_scale are retained for strict official
        # checkpoint coverage but are absent from this feature-only forward.
        # Leave only the two executed modules trainable for unfrozen DDP runs.
        self.model.requires_grad_(False)
        self.model.pvcnn.requires_grad_(True)
        self.model.triplane_transformer.requires_grad_(True)

    def forward(self, points):
        features = self.model.pvcnn(points, points)
        planes = self.model.triplane_transformer(features)
        return self.sample(planes[:, :, 64:], points)

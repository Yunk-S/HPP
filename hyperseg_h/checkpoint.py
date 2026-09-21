"""Audited loading, including official DDP-prefixed component checkpoints."""
import torch


def audit_load(module, state, name='module', minimum=0.95):
    module = getattr(module, 'module', module)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    expected = module.state_dict()
    matched = {k: v for k, v in state.items() if k in expected and v.shape == expected[k].shape}
    parameters = dict(module.named_parameters())
    total = sum(v.numel() for v in parameters.values())
    loaded = sum(v.numel() for k, v in parameters.items() if k in matched)
    ratio = loaded / max(total, 1)
    report = {'name': name, 'ratio': ratio, 'loaded_numel': loaded, 'total_numel': total,
              'missing': sorted(set(expected) - set(matched)),
              'unexpected': sorted(set(state) - set(expected)),
              'shape_mismatch': sorted(k for k in state if k in expected and state[k].shape != expected[k].shape)}
    if ratio < minimum or report['shape_mismatch']:
        raise RuntimeError(f'{name} checkpoint audit failed: {report}')
    module.load_state_dict(matched, strict=False)
    return report


def load_official(model, checkpoint):
    """Initialize old modules only; the new gate is intentionally freshly initialized."""
    return [audit_load(getattr(model, attr), checkpoint[key], attr) for attr, key in (
        ('enhancer', 'point_feature_enhancer_state_dict'),
        ('decoder', 'decoder_state_dict'), ('seg_head', 'seg_head_state_dict'))]


def save_checkpoint(path, model, config, **extra):
    torch.save({'format': 'hyperseg-h-v1', 'config': dict(config),
                'model_state_dict': model.state_dict(), **extra}, path)


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') == 'hyperseg-h-v1':
        # Full checkpoints must restore every new parameter, not just the large legacy modules.
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return [{'name': 'full_model', 'ratio': 1.0}]
    return load_official(model, checkpoint)

#!/usr/bin/env python3
"""Numerically compare HPP's official mode with a pinned S²AM3D checkout.

This intentionally takes an external checkout instead of making HPP point to
the upstream repository.  It prevents a self-comparison from being mistaken
for a compatibility regression test.
"""
import argparse
import subprocess
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PINNED_UPSTREAM_COMMIT = 'f1e1faf1030dacbc9dfb0be70d74afb45f56223f'


def load_source(path, name):
    module = types.ModuleType(name)
    module.__package__ = 'decoder.models'
    source = Path(path).read_text()
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def compare(upstream_root):
    upstream_root = Path(upstream_root).resolve()
    actual_commit = subprocess.check_output(
        ['git', '-C', str(upstream_root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_commit != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError(
            f'Upstream checkout is {actual_commit}; expected pinned {PINNED_UPSTREAM_COMMIT}')

    from decoder.models.CrossAttentionDecoder import Decoder
    from decoder.models.PointFeatureEnhancer import PointFeatureEnhancer
    from decoder.models.seghead_mlp import SegHead

    checks = []
    enhancer_args = dict(feature_dim=12, feature_proj_dim=24, pos_num_feats=8,
                         transformer_hidden_dim=24, transformer_num_heads=4,
                         transformer_num_layers=2, dropout=0.)
    upstream_enhancer = load_source(
        upstream_root / 'decoder/models/PointFeatureEnhancer.py', 'upstream_enhancer')
    reference = upstream_enhancer.PointFeatureEnhancer(**enhancer_args).eval()
    current = PointFeatureEnhancer(**enhancer_args, compatibility_mode='official-compatibility').eval()
    current.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(7)
    inputs = (torch.randn(2, 13, 12), torch.randn(2, 13, 3), None, torch.tensor([.1, .8]))
    checks.append(('PointFeatureEnhancer', (current(*inputs) - reference(*inputs)).abs().max().item()))

    upstream_decoder = load_source(
        upstream_root / 'decoder/models/CrossAttentionDecoder.py', 'upstream_decoder')
    reference = upstream_decoder.Decoder(24, 4, num_layers=3).eval()
    current = Decoder(24, 4, num_layers=3, gate_aware=False).eval()
    current.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(11)
    decoder_inputs = (torch.randn(2, 13, 24), torch.randn(2, 1, 24))
    checks.append(('CrossAttentionDecoder',
                   (current(*decoder_inputs) - reference(*decoder_inputs)).abs().max().item()))
    gated = Decoder(24, 4, num_layers=3, gate_aware=True,
                    prompt_propagation=False).eval()
    gated.load_state_dict(reference.state_dict(), strict=True)
    gates = torch.ones(2, 13)
    checks.append(('GateAwareDecoder(gates=1)',
                   (gated(decoder_inputs[0], decoder_inputs[1], gates=gates) -
                    reference(*decoder_inputs)).abs().max().item()))

    upstream_seghead = load_source(
        upstream_root / 'decoder/models/seghead_mlp.py', 'upstream_seghead')
    reference = upstream_seghead.SegHead(24, dropout=0.).eval()
    current = SegHead(24, dropout=0.).eval()
    current.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(13)
    seg_inputs = torch.randn(2, 13, 24)
    checks.append(('SegHead', (current(seg_inputs) - reference(seg_inputs)).abs().max().item()))
    for name, error in checks:
        if error > 1e-6:
            raise AssertionError(f'{name} compatibility error={error:.8g}')
    print(f'Upstream commit {actual_commit}: {len(checks)} compatibility checks passed')
    for name, error in checks:
        print(f'  {name}: max_abs_error={error:.8g}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', required=True,
                        help='Checkout of sumuru789/S2AM3D at the pinned commit')
    args = parser.parse_args(argv)
    compare(args.upstream_root)


if __name__ == '__main__':
    main()

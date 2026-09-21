#!/usr/bin/env python3
"""Audit official component or full HyperSeg-H checkpoints by parameter numel."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperseg_h.cli import load_config, make_model
from hyperseg_h.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', help='Required for official component checkpoints')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', help='Optional JSON audit report')
    args = parser.parse_args()
    stored = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = load_config(args.config) if args.config else stored.get('config')
    if not config:
        parser.error('Official checkpoints require --config')
    if stored.get('format') == 'hyperseg-h-v1' and stored['config']['model'] != config['model']:
        raise ValueError('Full checkpoint/config model settings differ')
    model = make_model(config, args.device)
    report = load_checkpoint(args.checkpoint, model)
    result = {'checkpoint': args.checkpoint, 'modules': report,
              'new_gate_initialization': stored.get('format') != 'hyperseg-h-v1'}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

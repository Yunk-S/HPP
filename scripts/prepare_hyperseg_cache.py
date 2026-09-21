#!/usr/bin/env python3
"""Convert aligned point/feature/mask arrays into unique-node uint8 caches.

Input JSON list:
  {model_id, points_path, features_path(optional), node_masks: {node: mask.npy},
   chains: [{node_ids: [root,...,leaf]}]}
Or --scale-labels: {model_id, data_path, features_path(optional)}, where data_path
is an official trusted .npy dictionary containing coord[N,3], label[N].
Hierarchy is never fabricated from flat labels.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyperseg_h.data import validate_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale-labels', action='store_true')
    args = parser.parse_args()
    source = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads(source.read_text())
    result, seen = [], set()
    for record in records:
        model_id = str(record['model_id'])
        if model_id in seen:
            raise ValueError(f'Duplicate model_id: {model_id}')
        seen.add(model_id)
        def load(field):
            return np.load(source.parent / record[field], allow_pickle=args.scale_labels)
        if args.scale_labels:
            original = load('data_path').item()
            points = original['coord']
            labels = np.asarray(original['label']).reshape(-1)
            if len(labels) != len(points):
                raise ValueError('Point/label length mismatch')
            masks = {str(i): torch.from_numpy((labels == i).astype(np.uint8)) for i in np.unique(labels)}
            chains = [{'node_ids': [i]} for i in masks]
        else:
            points = load('points_path')
            masks = {}
            for key, path in record['node_masks'].items():
                mask = np.load(source.parent / path, allow_pickle=False)
                if not np.isin(mask, [0, 1]).all():
                    raise ValueError(f'{key}: mask must be binary before conversion')
                masks[key] = torch.from_numpy(mask.astype(np.uint8))
            chains = record['chains']
        cache = {'model_id': model_id, 'points': torch.as_tensor(points, dtype=torch.float32),
                 'node_masks': masks, 'chains': chains}
        if record.get('features_path'):
            feat = load('features_path')
            if feat.dtype == object:
                feat = feat.item()['feat']
            cache['features'] = torch.as_tensor(feat, dtype=torch.float32)
        validate_cache(cache, hierarchy=not args.scale_labels)
        # File names are generated, never constructed from potentially path-like IDs.
        destination = output / f'object_{len(result):07d}.pt'
        if destination.exists():
            raise FileExistsError(destination)
        torch.save(cache, destination)
        result.append({'model_id': model_id, 'cache_path': destination.name})
    (output / 'manifest.json').write_text(json.dumps(result, indent=2))
    print(f'Created {len(result)} caches in {output}')


if __name__ == '__main__':
    main()

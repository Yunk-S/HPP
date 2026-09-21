#!/usr/bin/env python3
"""Build HyperSeg-H caches directly from PartNeXt Arrow annotations and GLBs.

The converter keeps the PartNeXt face-index convention intact:

    Arrow masks + hierarchyList + GLB
        -> sampled points + leaf masks
        -> union masks for internal nodes
        -> root-to-leaf chains consumable by prepare_hyperseg_cache.py

The dataset and meshes are intentionally external inputs; this script never
copies them into the repository.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _json_value(value, field):
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a JSON string or decoded object')
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {field}') from exc


def _iter_records(path):
    path = Path(path)
    # PartNeXt distributes annotations as Arrow shards. Do not treat every
    # JSON metadata file in a shard directory as an annotation table
    # (dataset_info.json is not record-shaped). JSON/JSONL remain supported
    # when passed explicitly via --annotations.
    paths = sorted(path.glob('*.arrow')) if path.is_dir() else [path]
    for source in paths:
        if source.suffix == '.jsonl':
            for line in source.read_text().splitlines():
                if line.strip():
                    yield json.loads(line)
            continue
        if source.suffix == '.json':
            data = json.loads(source.read_text())
            yield from (data if isinstance(data, list) else data['records'])
            continue
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise RuntimeError('Reading PartNeXt Arrow files requires pyarrow') from exc
        with pa.memory_map(str(source), 'r') as mapped:
            try:
                reader = ipc.open_file(mapped)
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            except pa.ArrowInvalid:
                mapped.seek(0)
                reader = ipc.open_stream(mapped)
                batches = iter(reader)
            for batch in batches:
                yield from batch.to_pylist()


def _find_mesh(mesh_root, model_id):
    root = Path(mesh_root)
    candidates = [
        root / 'glbs' / f'{model_id[:3]}-{model_id[3:6]}' / f'{model_id}.glb',
        root / 'glbs' / f'{model_id}.glb',
        root / f'{model_id}.glb',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.glob(f'**/{model_id}.glb'))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f'Could not locate GLB for model_id={model_id}')
    raise RuntimeError(f'Multiple GLBs found for model_id={model_id}: {matches[:4]}')


def _load_mesh_parts(path):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError('Mesh conversion requires trimesh') from exc
    loaded = trimesh.load(str(path), force='scene', process=False)
    if isinstance(loaded, trimesh.Trimesh):
        parts = [loaded]
    else:
        parts = [mesh for mesh in loaded.dump(concatenate=False)
                 if isinstance(mesh, trimesh.Trimesh)]
    if not parts:
        raise ValueError(f'No triangular mesh geometry found in {path}')
    return parts, trimesh.util.concatenate(parts)


def _sample_points(mesh, num_points, seed):
    import trimesh
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    finally:
        np.random.set_state(state)
    return points.astype(np.float32), face_indices.astype(np.int64)


def _face_masks(annotation, parts, face_indices):
    face_counts = _json_value(annotation['mesh_face_num'], 'mesh_face_num')
    counts = [int(face_counts[str(i)] if str(i) in face_counts else face_counts[i])
              for i in range(len(parts))]
    actual = [len(part.faces) for part in parts]
    if counts != actual:
        raise ValueError(f"mesh_face_num mismatch: annotation={counts}, mesh={actual}")
    offsets = np.cumsum([0] + actual[:-1]).tolist()
    masks = _json_value(annotation['masks'], 'masks')
    leaf_masks = {}
    for mask_id, mesh_faces in masks.items():
        global_faces = []
        for mesh_id, faces in mesh_faces.items():
            mesh_index = int(mesh_id)
            if mesh_index < 0 or mesh_index >= len(offsets):
                raise ValueError(f'Unknown mesh index {mesh_id} in mask {mask_id}')
            face_array = np.asarray(faces, dtype=np.int64)
            if (face_array < 0).any() or (face_array >= actual[mesh_index]).any():
                raise ValueError(f'Out-of-range face in mask {mask_id}, mesh {mesh_id}')
            global_faces.append(face_array + offsets[mesh_index])
        selected = np.concatenate(global_faces) if global_faces else np.empty(0, dtype=np.int64)
        leaf_masks[str(mask_id)] = np.isin(face_indices, selected).astype(np.uint8)
    return leaf_masks


def _hierarchy_cache(annotation, leaf_masks, model_id, points):
    hierarchy = _json_value(annotation['hierarchyList'], 'hierarchyList')
    node_masks, chains = {}, []

    def visit(node, prefix):
        node_id = str(node['nodeId'])
        if node_id in node_masks:
            raise ValueError(f'Duplicate hierarchy nodeId: {node_id}')
        children = node.get('children') or []
        if children:
            child_paths = [visit(child, prefix + [node_id]) for child in children]
            child_masks = [node_masks[str(child['nodeId'])] for child in children]
            mask = np.maximum.reduce(child_masks)
            paths = [path for group in child_paths for path in group]
        elif 'maskId' in node:
            mask_id = str(node['maskId'])
            if mask_id not in leaf_masks:
                raise ValueError(f'Leaf {node_id} refers to missing maskId={mask_id}')
            mask = leaf_masks[mask_id]
            paths = [prefix + [node_id]]
        else:
            raise ValueError(f'Hierarchy node {node_id} has neither children nor maskId')
        node_masks[node_id] = mask.astype(np.uint8)
        return paths

    for root in hierarchy:
        chains.extend(visit(root, []))
    if not chains:
        raise ValueError(f'{model_id}: hierarchyList contains no root-to-leaf chain')
    for path in chains:
        if any(not node_masks[node].any() for node in path):
            raise ValueError(f'{model_id}: hierarchy chain contains an empty node: {path}')
    return {
        'model_id': model_id,
        'points': torch.from_numpy(points),
        'node_masks': {node: torch.from_numpy(mask) for node, mask in node_masks.items()},
        'chains': [{'node_ids': path,
                    'granularities': np.linspace(0, 1, len(path), dtype=np.float32).tolist()}
                   for path in chains],
    }


def convert_record(record, mesh_root, num_points, seed):
    model_id = str(record.get('model_id', '')).strip()
    if not model_id:
        raise ValueError('Every annotation record requires model_id')
    mesh_path = _find_mesh(mesh_root, model_id)
    parts, mesh = _load_mesh_parts(mesh_path)
    points, face_indices = _sample_points(mesh, num_points, seed)
    leaf_masks = _face_masks(record, parts, face_indices)
    return _hierarchy_cache(record, leaf_masks, model_id, points)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, help='PartNeXt Arrow shard, JSON/JSONL, or directory')
    parser.add_argument('--mesh-root', required=True, help='PartNeXt_mesh root containing glbs/')
    parser.add_argument('--output', required=True, help='Cache output directory')
    parser.add_argument('--num-points', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-id', action='append', default=[])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    if args.num_points < 1:
        parser.error('--num-points must be positive')
    selected = set(args.model_id)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, seen = [], set()
    for record_index, record in enumerate(_iter_records(args.annotations)):
        model_id = str(record.get('model_id', '')).strip()
        if selected and model_id not in selected:
            continue
        if model_id in seen:
            raise ValueError(f'Duplicate model_id: {model_id}')
        seen.add(model_id)
        cache = convert_record(record, args.mesh_root, args.num_points, args.seed + record_index)
        from hyperseg_h.data import validate_cache
        validate_cache(cache, hierarchy=True)
        destination = output / f'object_{len(manifest):07d}.pt'
        torch.save(cache, destination)
        manifest.append({'model_id': model_id, 'cache_path': destination.name})
        if args.limit is not None and len(manifest) >= args.limit:
            break
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Created {len(manifest)} PartNeXt caches in {output}')


if __name__ == '__main__':
    main()

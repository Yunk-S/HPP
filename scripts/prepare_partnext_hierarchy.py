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


def _annotation_face_sets(annotation, parts):
    face_counts = _json_value(annotation['mesh_face_num'], 'mesh_face_num')
    counts = [int(face_counts[str(i)] if str(i) in face_counts else face_counts[i])
              for i in range(len(parts))]
    actual = [len(part.faces) for part in parts]
    if counts != actual:
        raise ValueError(f"mesh_face_num mismatch: annotation={counts}, mesh={actual}")
    offsets = np.cumsum([0] + actual[:-1]).tolist()
    masks = _json_value(annotation['masks'], 'masks')
    face_sets = {}
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
        face_sets[str(mask_id)] = (np.concatenate(global_faces)
                                   if global_faces else np.empty(0, dtype=np.int64))
    return face_sets


def _face_masks(annotation, parts, face_indices):
    face_sets = _annotation_face_sets(annotation, parts)
    return {mask_id: np.isin(face_indices, selected).astype(np.uint8)
            for mask_id, selected in face_sets.items()}


def _sample_face_points(mesh, face_ids, num_points, rng):
    """Area-weighted barycentric samples restricted to global face IDs."""
    face_ids = np.asarray(face_ids, dtype=np.int64)
    if face_ids.size == 0:
        raise ValueError('Cannot sample from an empty face set')
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)[face_ids]
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    total_area = float(areas.sum())
    probabilities = None if total_area <= 0 else areas / total_area
    selected = rng.choice(len(face_ids), size=num_points, replace=True, p=probabilities)
    r1 = np.sqrt(rng.random(num_points))
    r2 = rng.random(num_points)
    tri = triangles[selected]
    points = ((1 - r1)[:, None] * tri[:, 0] +
              (r1 * (1 - r2))[:, None] * tri[:, 1] +
              (r1 * r2)[:, None] * tri[:, 2])
    return points.astype(np.float32), face_ids[selected]


def _hierarchy_leaf_mask_ids(annotation):
    hierarchy = _json_value(annotation['hierarchyList'], 'hierarchyList')
    result = []

    def visit(node):
        children = node.get('children') or []
        if children:
            for child in children:
                visit(child)
        elif 'maskId' in node:
            mask_id = str(node['maskId'])
            if mask_id not in result:
                result.append(mask_id)
        else:
            raise ValueError(f"Hierarchy node {node.get('nodeId')} has neither children nor maskId")

    for root in hierarchy:
        visit(root)
    if not result:
        raise ValueError('hierarchyList contains no leaf maskId')
    return result


def _hierarchy_cache(annotation, leaf_masks, model_id, points):
    hierarchy = _json_value(annotation['hierarchyList'], 'hierarchyList')
    node_masks, node_support, chains = {}, {}, []

    def visit(node, prefix):
        node_id = str(node['nodeId'])
        if node_id in node_masks:
            raise ValueError(f'Duplicate hierarchy nodeId: {node_id}')
        children = node.get('children') or []
        if children:
            child_results = [visit(child, prefix + [node_id]) for child in children]
            child_paths = [result[0] for result in child_results]
            child_masks = [node_masks[str(child['nodeId'])] for child in children]
            mask = np.maximum.reduce(child_masks)
            paths = [path for group in child_paths for path in group]
            support = frozenset().union(*(result[1] for result in child_results))
        elif 'maskId' in node:
            mask_id = str(node['maskId'])
            if mask_id not in leaf_masks:
                raise ValueError(f'Leaf {node_id} refers to missing maskId={mask_id}')
            mask = leaf_masks[mask_id]
            paths = [prefix + [node_id]]
            support = frozenset((mask_id,))
        else:
            raise ValueError(f'Hierarchy node {node_id} has neither children nor maskId')
        node_masks[node_id] = mask.astype(np.uint8)
        node_support[node_id] = support
        return paths, support

    for root in hierarchy:
        chains.extend(visit(root, [])[0])
    if not chains:
        raise ValueError(f'{model_id}: hierarchyList contains no root-to-leaf chain')
    for path in chains:
        if any(not node_masks[node].any() for node in path):
            raise ValueError(f'{model_id}: hierarchy chain contains an empty node: {path}')
    contracted_chains = []
    for path in chains:
        contracted, groups = [], []
        for node_id in path:
            if contracted and node_support[node_id] == node_support[contracted[-1]]:
                groups[-1].append(node_id)
            else:
                contracted.append(node_id)
                groups.append([node_id])
        contracted_chains.append({
            'node_ids': contracted,
            'granularities': np.linspace(0, 1, len(contracted), dtype=np.float32).tolist(),
            'original_node_ids': path,
            'contracted_node_ids': contracted,
            'contraction_groups': groups,
        })
    return {
        'model_id': model_id,
        'points': torch.from_numpy(points),
        'node_masks': {node: torch.from_numpy(mask) for node, mask in node_masks.items()},
        'chains': contracted_chains,
    }


def _sample_points(mesh, face_sets, leaf_ids, num_points, seed, leaf_quota):
    if leaf_quota < 0:
        raise ValueError('leaf_quota must be non-negative')
    if leaf_quota and len(leaf_ids) * leaf_quota > num_points:
        raise ValueError(
            f'leaf quota requires {len(leaf_ids) * leaf_quota} points, exceeds num_points={num_points}')
    rng = np.random.default_rng(seed)
    points, face_indices = [], []
    for leaf_id in leaf_ids:
        sampled_points, sampled_faces = _sample_face_points(
            mesh, face_sets[leaf_id], leaf_quota, rng)
        points.append(sampled_points)
        face_indices.append(sampled_faces)
    remaining = num_points - len(points) * leaf_quota
    all_faces = np.arange(len(mesh.faces), dtype=np.int64)
    if remaining:
        sampled_points, sampled_faces = _sample_face_points(mesh, all_faces, remaining, rng)
        points.append(sampled_points)
        face_indices.append(sampled_faces)
    points = np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=np.float32)
    face_indices = np.concatenate(face_indices, axis=0) if face_indices else np.empty(0, dtype=np.int64)
    order = rng.permutation(len(points))
    return points[order], face_indices[order]


def convert_record(record, mesh_root, num_points, seed, leaf_quota=10):
    model_id = str(record.get('model_id', '')).strip()
    if not model_id:
        raise ValueError('Every annotation record requires model_id')
    mesh_path = _find_mesh(mesh_root, model_id)
    parts, mesh = _load_mesh_parts(mesh_path)
    face_sets = _annotation_face_sets(record, parts)
    leaf_ids = _hierarchy_leaf_mask_ids(record)
    missing = [leaf_id for leaf_id in leaf_ids if leaf_id not in face_sets]
    if missing:
        raise ValueError(f'Missing face masks for hierarchy leaves: {missing[:8]}')
    points, face_indices = _sample_points(
        mesh, face_sets, leaf_ids, num_points, seed, leaf_quota)
    leaf_masks = _face_masks(record, parts, face_indices)
    return _hierarchy_cache(record, leaf_masks, model_id, points)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, help='PartNeXt Arrow shard, JSON/JSONL, or directory')
    parser.add_argument('--mesh-root', required=True, help='PartNeXt_mesh root containing glbs/')
    parser.add_argument('--output', required=True, help='Cache output directory')
    parser.add_argument('--num-points', type=int, default=10000)
    parser.add_argument('--leaf-quota', type=int, default=10,
                        help='Guaranteed samples per hierarchy leaf before global area sampling')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-id', action='append', default=[])
    parser.add_argument('--limit', type=int)
    parser.add_argument('--strict', action='store_true',
                        help='Abort on the first invalid annotation instead of recording a rejection')
    args = parser.parse_args(argv)
    if args.num_points < 1 or args.leaf_quota < 0:
        parser.error('--num-points must be positive and --leaf-quota non-negative')
    selected = set(args.model_id)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, seen, rejected = [], set(), []
    for record_index, record in enumerate(_iter_records(args.annotations)):
        model_id = str(record.get('model_id', '')).strip()
        if selected and model_id not in selected:
            continue
        try:
            if not model_id:
                raise ValueError('missing model_id')
            if model_id in seen:
                raise ValueError(f'duplicate model_id: {model_id}')
            cache = convert_record(record, args.mesh_root, args.num_points,
                                   args.seed + record_index, args.leaf_quota)
            from hyperseg_h.data import validate_cache
            validate_cache(cache, hierarchy=True)
            destination = output / f'object_{len(manifest):07d}.pt'
            torch.save(cache, destination)
            manifest.append({'model_id': model_id, 'cache_path': destination.name})
            seen.add(model_id)
        except Exception as exc:
            rejection = {'record_index': record_index, 'model_id': model_id,
                         'reason': f'{type(exc).__name__}: {exc}'}
            rejected.append(rejection)
            if args.strict:
                raise
        if args.limit is not None and len(manifest) >= args.limit:
            break
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'accepted_ids.json').write_text(json.dumps(
        [item['model_id'] for item in manifest], indent=2))
    (output / 'rejected_ids.json').write_text(json.dumps(
        [item['model_id'] for item in rejected], indent=2))
    (output / 'rejected_reasons.json').write_text(json.dumps(rejected, indent=2))
    print(f'Created {len(manifest)} PartNeXt caches in {output}; rejected {len(rejected)}')


if __name__ == '__main__':
    main()

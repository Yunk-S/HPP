"""Unique-node uint8 caches, ancestor chains, and reproducible evaluation sampling."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


def unit_sphere(points):
    points = points.float()
    centered = points - points.mean(dim=-2, keepdim=True)
    return centered / centered.norm(dim=-1, keepdim=True).amax(dim=-2, keepdim=True).clamp_min(1e-8)


def official_encoder_normalize(points):
    """Match the bundled PartField/S²AM3D encoder preprocessing."""
    points = points.float()
    bbmin, bbmax = points.amin(dim=-2, keepdim=True), points.amax(dim=-2, keepdim=True)
    center = (bbmin + bbmax) * 0.5
    extent = (bbmax - bbmin).amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return (points - center) * (2.0 * 0.9 / extent)


def official_decoder_normalize(points):
    """Match the legacy decoder's per-coordinate standardization."""
    points = points.float()
    return (points - points.mean(dim=-2, keepdim=True)) / (points.std(dim=-2, keepdim=True) + 1e-6)


def normalize_points(points, mode='unit-sphere'):
    if mode == 'unit-sphere':
        return unit_sphere(points)
    if mode in ('official-decoder', 'legacy-standardize'):
        return official_decoder_normalize(points)
    if mode in ('official-encoder', 'partfield-encoder'):
        return official_encoder_normalize(points)
    raise ValueError(f'Unknown normalization mode: {mode}')


def boundary_prompt(points, mask, alpha=0.5, chunk_size=1024):
    """Official center/interior score, with bounded-memory nearest-background distance."""
    foreground = torch.where(mask.bool())[0]
    background = torch.where(~mask.bool())[0]
    if foreground.numel() == 0:
        raise ValueError('Cannot select a prompt from an empty target')
    fg = points[foreground].float()
    center = (fg - fg.mean(0)).norm(dim=-1)
    distance = torch.full_like(center, 1e6)
    if background.numel():
        for start in range(0, len(foreground), chunk_size):
            nearest = torch.full_like(center[start:start + chunk_size], float('inf'))
            for other in range(0, len(background), chunk_size):
                d = torch.cdist(fg[start:start + chunk_size], points[background[other:other + chunk_size]].float())
                nearest = torch.minimum(nearest, d.amin(-1))
            distance[start:start + chunk_size] = nearest
    center = (center - center.min()) / (center.max() - center.min() + 1e-6)
    distance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
    return foreground[(alpha * (1 - center) + (1 - alpha) * distance).argmax()]


def leaf_stratified_sample(node_masks, leaf_ids, num_points, rng, n_min=10):
    n = next(iter(node_masks.values())).numel()
    if n == 0 or num_points <= 0:
        raise ValueError('Point pool and num_points must be positive')
    selected = set()
    for leaf_id in leaf_ids:
        candidates = torch.where(node_masks[leaf_id].bool())[0].cpu().numpy()
        selected.update(rng.choice(candidates, min(n_min, len(candidates)), replace=False).tolist())
    if len(selected) > num_points:
        raise ValueError('Leaf quotas exceed num_points; increase budget or reduce n_min')
    indices = np.array(sorted(selected), dtype=np.int64)
    remaining = np.setdiff1d(np.arange(n), indices)
    take = min(num_points - len(indices), len(remaining))
    indices = np.concatenate([indices, rng.choice(remaining, take, replace=False)])
    if len(indices) < num_points:
        indices = np.concatenate([indices, rng.choice(np.arange(n), num_points - len(indices), replace=True)])
    rng.shuffle(indices)
    return torch.from_numpy(indices)


def _uniform_sample_indices(num_available, num_points, rng):
    if num_available == 0 or num_points <= 0:
        raise ValueError('Point pool and num_points must be positive')
    return torch.from_numpy(rng.choice(
        num_available, num_points, replace=num_available < num_points).astype(np.int64))


def cache_leaf_ids(cache):
    """Return unique target leaves in stable chain order."""
    return list(dict.fromkeys(chain['node_ids'][-1] for chain in cache['chains']))


def read_manifest(path):
    path = Path(path)
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError('Manifest must be a JSON list of {model_id, cache_path}')
    for record in records:
        if not str(record.get('model_id', '')).strip() or 'cache_path' not in record:
            raise ValueError('Every record requires an explicit model_id and cache_path')
        record['model_id'] = str(record['model_id'])
        record['cache_path'] = str((path.parent / record['cache_path']).resolve())
    return records


def exclude_overlap(train_records, test_records, output_dir=None):
    test_ids = {str(r['model_id']) for r in test_records}
    overlap = sorted({str(r['model_id']) for r in train_records} & test_ids)
    kept = [r for r in train_records if str(r['model_id']) not in test_ids]
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'excluded_overlap_ids.txt').write_text(''.join(x + '\n' for x in overlap))
        (out / 'train_deoverlapped.json').write_text(json.dumps(kept, indent=2))
    return kept, overlap


def validate_cache(cache, hierarchy=True):
    points, masks = cache['points'], cache['node_masks']
    if points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all() or len(points) == 0:
        raise ValueError('points must be finite [N,3]')
    if not masks:
        raise ValueError('node_masks must be nonempty')
    for key, mask in masks.items():
        if mask.dtype != torch.uint8 or mask.shape != (len(points),) or (mask > 1).any():
            raise ValueError(f'node {key}: masks must be binary uint8[N]')
    if 'features' in cache and (cache['features'].ndim != 2 or len(cache['features']) != len(points)
                                or not torch.isfinite(cache['features']).all()):
        raise ValueError('features must be finite, aligned [N,C]')
    if not cache.get('chains'):
        raise ValueError('cache requires chains')
    for chain in cache['chains']:
        ids = chain['node_ids']
        if not ids or len(set(ids)) != len(ids) or any(i not in masks for i in ids):
            raise ValueError('chain must contain distinct existing node_ids')
        if hierarchy and len(ids) < 2:
            raise ValueError('True hierarchy requires at least two annotated levels')
        if any(not masks[i].any() for i in ids):
            raise ValueError('Empty annotated node')
        for parent, child in zip(ids, ids[1:]):
            if (masks[child].bool() & ~masks[parent].bool()).any():
                raise ValueError('Child mask is not contained in parent')
        expected = torch.linspace(0, 1, len(ids)) if len(ids) > 1 else torch.zeros(1)
        if 'granularities' in chain and not torch.allclose(torch.as_tensor(chain['granularities']).float(), expected):
            raise ValueError('granularities must be path-relative k/(K-1)')


class HierarchyDataset(Dataset):
    def __init__(self, records, num_points=10000, hierarchy=True, training=False, seed=0,
                 leaf_quota=10, normalization='unit-sphere', backbone_normalization=None,
                 prompt_normalization='unit-sphere'):
        self.records, self.num_points, self.hierarchy = records, num_points, hierarchy
        self.training, self.seed, self.leaf_quota = training, seed, leaf_quota
        self.normalization = normalization
        self.backbone_normalization = backbone_normalization
        self.prompt_normalization = prompt_normalization
        self.items, self.object_ids = [], []
        for obj, record in enumerate(records):
            cache = torch.load(record['cache_path'], map_location='cpu', weights_only=True)
            validate_cache(cache, hierarchy)
            if str(cache.get('model_id', record['model_id'])) != record['model_id']:
                raise ValueError('Manifest/cache model_id mismatch')
            for chain in range(len(cache['chains'])):
                self.items.append((obj, chain))
                self.object_ids.append(record['model_id'])
        if not self.items:
            raise ValueError('No examples remain after ID filtering')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        obj, chain_idx = self.items[index]
        record = self.records[obj]
        cache = torch.load(record['cache_path'], map_location='cpu', weights_only=True)
        rng = np.random.default_rng(int(np.random.randint(2**31)) if self.training else self.seed + obj)
        masks = cache['node_masks']
        if self.hierarchy:
            leaf_ids = list(dict.fromkeys(chain['node_ids'][-1] for chain in cache['chains']))
            indices = leaf_stratified_sample(masks, leaf_ids, self.num_points, rng, self.leaf_quota)
        else:
            # A1 uses uniform point sampling; hierarchy quotas would alter the benchmark.
            indices = _uniform_sample_indices(len(cache['points']), self.num_points, rng)
        raw_points = cache['points'][indices].float()
        points = normalize_points(raw_points, self.normalization)
        prompt_points = normalize_points(raw_points, self.prompt_normalization)
        ids = cache['chains'][chain_idx]['node_ids']
        labels = torch.stack([masks[i][indices].float() for i in ids])
        valid = labels.bool().any(-1)
        if not valid.any():
            raise ValueError(f'{record["model_id"]}: all targets vanished during sampling')
        deepest = torch.where(valid)[0][-1]
        prompt = boundary_prompt(prompt_points, labels[deepest])
        out = {'points': points, 'labels': labels, 'valid': valid,
               'granularities': torch.linspace(0, 1, len(ids)) if len(ids) > 1 else torch.zeros(1),
               'scales': labels.mean(-1), 'prompt_indices': prompt,
               'model_id': record['model_id'], 'node_ids': ids}
        if self.backbone_normalization is not None:
            out['backbone_points'] = normalize_points(raw_points, self.backbone_normalization)
        if 'features' in cache:
            out['features'] = cache['features'][indices].float()
        return out


class A1ObjectDataset(Dataset):
    """S²AM3D-style A1 sampling: one object, then one visible target part.

    Training samples one object per item, uniformly subsamples its point pool,
    and only then chooses a random target leaf that remains represented.  Eval
    expands each object into deterministic object/target pairs while reusing
    the same sampled point set for all targets of that object.
    """
    def __init__(self, records, num_points=10000, training=False, seed=0,
                 normalization='unit-sphere', backbone_normalization=None,
                 prompt_normalization='unit-sphere'):
        self.records, self.num_points = records, num_points
        self.training, self.seed = training, seed
        self.normalization = normalization
        self.backbone_normalization = backbone_normalization
        self.prompt_normalization = prompt_normalization
        self.items, self.object_ids = [], []
        self.leaf_ids = []
        for obj, record in enumerate(records):
            cache = torch.load(record['cache_path'], map_location='cpu', weights_only=True)
            validate_cache(cache, hierarchy=False)
            if str(cache.get('model_id', record['model_id'])) != record['model_id']:
                raise ValueError('Manifest/cache model_id mismatch')
            leaves = cache_leaf_ids(cache)
            if not leaves:
                raise ValueError(f'{record["model_id"]}: cache has no target leaves')
            self.leaf_ids.append(leaves)
            if training:
                self.items.append((obj, None))
                self.object_ids.append(record['model_id'])
            else:
                for leaf_id in leaves:
                    self.items.append((obj, leaf_id))
                    self.object_ids.append(record['model_id'])
        if not self.items:
            raise ValueError('No A1 objects remain after ID filtering')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        obj, requested_leaf = self.items[index]
        record = self.records[obj]
        cache = torch.load(record['cache_path'], map_location='cpu', weights_only=True)
        rng = np.random.default_rng(
            int(np.random.randint(2**31)) if self.training else self.seed + obj)
        indices = _uniform_sample_indices(len(cache['points']), self.num_points, rng)
        raw_points = cache['points'][indices].float()
        points = normalize_points(raw_points, self.normalization)
        prompt_points = normalize_points(raw_points, self.prompt_normalization)
        visible = [leaf_id for leaf_id in self.leaf_ids[obj]
                   if cache['node_masks'][leaf_id][indices].any()]
        if self.training:
            if not visible:
                raise ValueError(f'{record["model_id"]}: no target leaf survived sampling')
            target_id = visible[int(rng.integers(len(visible)))]
        else:
            target_id = requested_leaf
            if target_id not in visible:
                raise ValueError(
                    f'{record["model_id"]}: target leaf {target_id} vanished during A1 sampling')
        labels = cache['node_masks'][target_id][indices].float().unsqueeze(0)
        prompt = boundary_prompt(prompt_points, labels[0])
        out = {
            'points': points,
            'labels': labels,
            'valid': torch.ones(1, dtype=torch.bool),
            'granularities': torch.zeros(1),
            'scales': labels.mean(-1),
            'prompt_indices': prompt,
            'model_id': record['model_id'],
            'node_ids': [target_id],
        }
        if self.backbone_normalization is not None:
            out['backbone_points'] = normalize_points(raw_points, self.backbone_normalization)
        if 'features' in cache:
            out['features'] = cache['features'][indices].float()
        return out


class ObjectBalancedSampler(WeightedRandomSampler):
    def __init__(self, object_ids, generator=None):
        from collections import Counter
        counts = Counter(object_ids)
        super().__init__([1.0 / counts[i] for i in object_ids], len(object_ids), replacement=True, generator=generator)


class DistributedObjectBalancedSampler(Sampler):
    """One random item per object per epoch, globally shuffled then sharded.

    A1 selects objects (the dataset then selects a visible target); A2/C selects
    one chain uniformly within each object. No object appears on multiple ranks
    or twice in an epoch. Drop the shuffled tail to a full global batch instead
    of padding with repeated objects. The tail changes with seed + epoch.
    """
    def __init__(self, object_ids, num_replicas=1, rank=0, batch_size=1, seed=0):
        if num_replicas < 1 or not 0 <= rank < num_replicas or batch_size < 1:
            raise ValueError('Invalid sampler rank, world size or batch size')
        self.groups = {}
        for index, object_id in enumerate(object_ids):
            self.groups.setdefault(object_id, []).append(index)
        self.num_replicas, self.rank = num_replicas, rank
        self.seed, self.epoch = seed, 0
        global_batch = num_replicas * batch_size
        self.total_size = len(self.groups) // global_batch * global_batch
        self.num_samples = self.total_size // num_replicas
        self.dropped_objects = len(self.groups) - self.total_size
        if self.total_size == 0:
            raise ValueError('Need at least one global batch of distinct objects; reduce batch size or ranks')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups = list(self.groups.values())
        order = torch.randperm(len(groups), generator=generator).tolist()[:self.total_size]
        indices = [groups[i][torch.randint(len(groups[i]), (1,), generator=generator).item()]
                   for i in order]
        return iter(indices[self.rank::self.num_replicas])


def collate_hierarchy(batch):
    levels = max(len(x['labels']) for x in batch)
    b, n = len(batch), len(batch[0]['points'])
    result = {'points': torch.stack([x['points'] for x in batch]),
              'prompt_indices': torch.stack([x['prompt_indices'] for x in batch]),
              'labels': torch.zeros(b, levels, n), 'valid': torch.zeros(b, levels, dtype=torch.bool),
              'granularities': torch.zeros(b, levels), 'scales': torch.zeros(b, levels),
              'model_id': [x['model_id'] for x in batch]}
    if all('backbone_points' in x for x in batch):
        result['backbone_points'] = torch.stack([x['backbone_points'] for x in batch])
    elif any('backbone_points' in x for x in batch):
        raise ValueError('Cannot mix records with and without backbone_points')
    if all('features' in x for x in batch):
        result['features'] = torch.stack([x['features'] for x in batch])
    elif any('features' in x for x in batch):
        raise ValueError('Cannot mix cached features and raw-point records')
    for i, item in enumerate(batch):
        for key in ('labels', 'valid', 'granularities', 'scales'):
            result[key][i, :len(item[key])] = item[key]
    return result

"""Track A1/A2/C train/eval and explicit object-ID audit CLI."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import HyperSegH
from .data import read_manifest, exclude_overlap, HierarchyDataset, ObjectBalancedSampler, collate_hierarchy
from .checkpoint import load_checkpoint, save_checkpoint


def load_config(path):
    if path is None:
        return {}
    import yaml
    return yaml.safe_load(Path(path).read_text())


def make_model(config, device, initialize_backbone=True):
    options = dict(config['model'])
    backbone = config.get('backbone')
    if backbone:
        from .backbone import OfficialBackbone
        options['backbone'] = OfficialBackbone(
            backbone['config'], backbone.get('checkpoint'),
            load_checkpoint=initialize_backbone)
        options['freeze_backbone'] = backbone.get('freeze', True)
    return HyperSegH(**options).to(device)


def validate_protocol(config, track, num_points):
    model = config['model']
    signal, hierarchy = model['control_signal'], model['hierarchy_enabled']
    if track == 'A1' and (hierarchy or signal not in ('scale', 'scale-proxy')):
        raise ValueError('A1 requires scale for legacy or scale-proxy for HyperSeg-H, without hierarchy')
    if track in ('A2', 'C') and model['model_variant'] != 'legacy' and (signal != 'hierarchy' or not hierarchy):
        raise ValueError('A2/C HyperSeg-H requires true hierarchy; scale proxies are forbidden')
    if track == 'C' and signal != 'hierarchy':
        raise ValueError('Track C g sweep requires explicit hierarchy control (including legacy ablation)')
    if config.get('normalization', 'unit-sphere') not in (
            'unit-sphere', 'official-decoder', 'legacy-standardize'):
        raise ValueError('Unknown decoder normalization for track CLI')
    if num_points != 10000:
        print('NON-BENCHMARK RUN: point count differs from the 10,000-point protocol')


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.7, sweep_steps=11, track='A2'):
    model.eval()
    rows, per_level, sweep_rows = [], {}, []
    containment_numerator = containment_denominator = 0
    violating_pairs = pair_count = 0
    consistent_chains = total_chains = 0
    covered = foreground_count = 0
    for raw in loader:
        batch = to_device(raw, device)
        controls = batch['granularities'] if model.control_signal == 'hierarchy' else batch['scales']
        features = batch.get('features')
        backbone_points = batch.get('backbone_points')
        if features is None and model.backbone is not None:
            features = model.backbone(backbone_points if backbone_points is not None else batch['points'])
        enhanced = (model.encode_features(batch['points'], features, backbone_points=backbone_points)
                    if model.encoder_is_control_invariant else None)
        masks = []
        for level in range(controls.shape[1]):
            if enhanced is None:
                probs, aux = model(batch['points'], batch['prompt_indices'], controls[:, level], features, True)
            else:
                probs, aux = model.decode_features(
                    enhanced, batch['prompt_indices'], controls[:, level], True)
            masks.append(probs > threshold)
            target = batch['labels'][:, level].bool()
            valid = batch['valid'][:, level]
            intersection = (masks[-1] & target).sum(-1)
            union = (masks[-1] | target).sum(-1)
            iou = intersection.float() / union.clamp_min(1)
            for b in torch.where(valid)[0].tolist():
                value = float(iou[b])
                per_level.setdefault(str(level), []).append(value)
                rows.append({'model_id': batch['model_id'][b], 'level': level, 'iou': value,
                             'control': float(controls[b, level]), 'prompt_index': int(batch['prompt_indices'][b])})
            if 'energy' in aux:
                positive = target & valid[:, None]
                covered += int(((aux['energy'] <= 1e-6) & positive).sum())
                foreground_count += int(positive.sum())
        predictions = torch.stack(masks, 1)
        for b in range(predictions.shape[0]):
            levels = torch.where(batch['valid'][b])[0].tolist()
            if len(levels) < 2:
                continue
            chain_ok = True
            for i, j in zip(levels, levels[1:]):
                fine, coarse = predictions[b, j], predictions[b, i]
                outside = fine & ~coarse
                containment_numerator += int(outside.sum())
                containment_denominator += int(fine.sum())
                violating_pairs += int(outside.any())
                pair_count += 1
                chain_ok = chain_ok and not bool(outside.any())
            consistent_chains += int(chain_ok)
            total_chains += 1
        if track == 'C':
            previous = None
            for g in torch.linspace(0, 1, sweep_steps, device=device):
                control = g.expand(batch['points'].shape[0])
                if enhanced is None:
                    probs, aux = model(batch['points'], batch['prompt_indices'], control, features, True)
                else:
                    probs, aux = model.decode_features(
                        enhanced, batch['prompt_indices'], control, True)
                mask = probs > threshold
                for b in range(len(mask)):
                    row = {'model_id': batch['model_id'][b], 'g': float(g), 'mask_fraction': float(mask[b].float().mean()),
                           'outside_previous': None if previous is None else int((mask[b] & ~previous[b]).sum())}
                    for key in ('rho', 'psi'):
                        if key in aux:
                            row[key] = float(aux[key][b])
                    sweep_rows.append(row)
                previous = mask
    if not rows:
        raise ValueError('No valid evaluation targets')
    by_object = {}
    for row in rows:
        by_object.setdefault(row['model_id'], []).append(row['iou'])
    object_mean = 100 * float(np.mean([np.mean(v) for v in by_object.values()]))
    return {'track': track, 'threshold': threshold,
            'target_mean_iou_percent': 100 * float(np.mean([r['iou'] for r in rows])),
            'object_mean_iou_percent': object_mean,
            'official_interactive_iou_percent': object_mean,
            'per_level_iou_percent': {k: 100 * float(np.mean(v)) for k, v in per_level.items()},
            'containment_violation_point_rate': containment_numerator / max(containment_denominator, 1) if pair_count else None,
            'containment_violation_pair_rate': violating_pairs / pair_count if pair_count else None,
            'same_point_ancestor_consistency': consistent_chains / total_chains if total_chains else None,
            'cone_coverage': covered / foreground_count if foreground_count else None,
            'targets': rows, 'sweep': sweep_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'eval', 'audit-ids', 'export'])
    parser.add_argument('--track', choices=['A1', 'A2', 'C'], default='A2')
    parser.add_argument('--config')
    parser.add_argument('--manifest')
    parser.add_argument('--test-manifest', action='append', default=[], help='Repeat for ALL held-out datasets')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-points', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.7, help='Fix using validation only')
    parser.add_argument('--sweep-steps', type=int, default=11)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--model-variant', choices=['legacy', 'hyperseg-h', 'radius-film', 'spherical-hierarchy'])
    parser.add_argument('--control-signal', choices=['scale', 'scale-proxy', 'hierarchy'])
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1 or args.sweep_steps < 2 or args.epochs < 1:
        parser.error('threshold must lie in (0,1), sweep_steps >= 2, epochs >= 1')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    if args.action == 'audit-ids':
        if not args.manifest or not args.test_manifest:
            parser.error('audit-ids requires --manifest and --test-manifest')
        test = [r for p in args.test_manifest for r in read_manifest(p)]
        kept, overlap = exclude_overlap(read_manifest(args.manifest), test, output)
        print(json.dumps({'kept': len(kept), 'excluded_ids': overlap}))
        return
    config = load_config(args.config)
    stored = None
    if not config:
        if not args.checkpoint:
            parser.error('Supply --config or a full HyperSeg-H --checkpoint')
        stored = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if 'config' not in stored:
            parser.error('Official checkpoints require an explicit model config')
        config = stored['config']
    config.setdefault('model', {})
    config['model'].setdefault('model_variant', 'hyperseg-h')
    config['model'].setdefault('compatibility_mode', 'official-compatibility')
    if args.model_variant:
        config['model']['model_variant'] = args.model_variant
    if args.control_signal:
        config['model']['control_signal'] = args.control_signal
        config['model']['hierarchy_enabled'] = args.control_signal == 'hierarchy'
    if 'control_signal' not in config['model'] or 'hierarchy_enabled' not in config['model']:
        parser.error('Config must explicitly set control_signal and hierarchy_enabled')
    validate_protocol(config, args.track, args.num_points)
    if args.checkpoint and stored is None:
        stored = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    initialize_backbone = not (stored is not None and stored.get('format') == 'hyperseg-h-v1')
    model = make_model(config, args.device, initialize_backbone=initialize_backbone)
    audit = []
    if args.checkpoint:
        stored = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if stored.get('format') == 'hyperseg-h-v1' and stored['config']['model'] != config['model']:
            raise ValueError('Full checkpoint model config differs: use its saved control and compatibility settings')
        audit = load_checkpoint(args.checkpoint, model)
        if stored.get('format') == 'hyperseg-h-v1':
            for key in ('training_object_ids', 'test_manifests', 'excluded_overlap_ids'):
                if key in stored['config']:
                    config[key] = stored['config'][key]
    elif args.action in ('eval', 'export'):
        parser.error('eval/export requires a trained --checkpoint')
    if args.action == 'export':
        output.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(output, model, config, audit=audit)
        print(f'Exported single checkpoint: {output}')
        return
    if not args.manifest:
        parser.error('train/eval requires --manifest')
    output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(args.manifest)
    if args.action == 'train':
        if not args.test_manifest:
            parser.error('Training requires --test-manifest for strict object-ID de-overlap')
        test = [r for p in args.test_manifest for r in read_manifest(p)]
        records, overlap = exclude_overlap(records, test, output)
        config['excluded_overlap_ids'] = overlap
        config['training_object_ids'] = sorted({r['model_id'] for r in records})
        config['test_manifests'] = args.test_manifest
    elif 'training_object_ids' in config:
        overlap = set(config['training_object_ids']) & {r['model_id'] for r in records}
        if overlap:
            raise ValueError(f'Evaluation IDs overlap checkpoint training IDs: {sorted(overlap)}')
    dataset = HierarchyDataset(
        records, args.num_points, hierarchy=args.track != 'A1',
        training=args.action == 'train', seed=args.seed,
        normalization=config.get('normalization', 'unit-sphere'),
        backbone_normalization=config.get('backbone_normalization'))
    if args.track == 'A1':
        for record in records:
            cache = torch.load(record['cache_path'], weights_only=True, map_location='cpu')
            if any(len(c['node_ids']) != 1 for c in cache['chains']):
                raise ValueError('A1 cache requires one singleton chain per target mask')
    sampler = ObjectBalancedSampler(dataset.object_ids) if args.action == 'train' else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_hierarchy)
    provenance = {'config': config, 'args': vars(args), 'checkpoint_audit': audit,
                  'protocol_status': 'implementation; official benchmark parity requires dataset/protocol verification'}
    (output / 'run_config.json').write_text(json.dumps(provenance, indent=2))
    if args.action == 'train':
        from training.decoder_train.code.train import HyperSegTrainer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_options = dict(config.get('loss', {}))
        if args.track == 'A1':
            loss_options.update(contain_weight=0., control_weight=0.)
        trainer = HyperSegTrainer(model, optimizer, args.device, args.track, args.amp, loss_options)
        history = []
        for epoch in range(args.epochs):
            for batch in loader:
                history.append({'epoch': epoch, **trainer.step(batch)})
            save_checkpoint(output / 'latest.pt', model, config, epoch=epoch + 1,
                            optimizer=optimizer.state_dict(), scaler=trainer.scaler.state_dict(), audit=audit)
            print(json.dumps(history[-1]), flush=True)
        (output / 'train_metrics.json').write_text(json.dumps(history, indent=2))
    else:
        metrics = evaluate(model, loader, args.device, args.threshold, args.sweep_steps, args.track)
        metrics['num_points'] = args.num_points
        metrics['protocol_status'] = provenance['protocol_status']
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: v for k, v in metrics.items() if k not in ('targets', 'sweep')}, indent=2))


if __name__ == '__main__':
    main()

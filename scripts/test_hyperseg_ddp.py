#!/usr/bin/env python3
"""CPU regression tests, including real two-process Gloo DDP (not a GPU test).

Run: python scripts/test_hyperseg_ddp.py
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)

from hyperseg_h.checkpoint import load_checkpoint, save_checkpoint
from hyperseg_h.data import DistributedObjectBalancedSampler
from hyperseg_h.distributed import (init_distributed,
    cleanup_distributed, wrap_model, unwrap_model, resolve_precision, autocast_dtype)
from hyperseg_h.model import HyperSegH
from training.decoder_train.code.train import HyperSegTrainer


def model_options(**kwargs):
    return dict(feature_dim=12, hidden_dim=24, hyper_dim=8, num_heads=4,
                enhancer_layers=1, decoder_layers=2, dropout=0.,
                control_signal='hierarchy', hierarchy_enabled=True,
                compatibility_mode='batch-first-corrected', **kwargs)


def batch(levels=3):
    labels = torch.zeros(1, levels, 12)
    for i in range(levels):
        labels[:, i, :12 - 3 * i] = 1
    return {'points': torch.randn(1, 12, 3), 'features': torch.randn(1, 12, 12),
            'prompt_indices': torch.tensor([0]), 'labels': labels,
            'valid': torch.ones(1, levels, dtype=torch.bool),
            'granularities': torch.linspace(0, 1, levels)[None],
            'scales': labels.mean(-1)}


def reference_predict(model, data, midpoints=True):
    """The pre-DDP trainer algorithm, independently exercised as a reference."""
    controls = data['granularities'] if model.control_signal == 'hierarchy' else data['scales']
    features = data.get('features')
    if features is None:
        features = model.backbone(data.get('backbone_points', data['points']))
    enhanced = model.encode_features(data['points'], features) if model.encoder_is_control_invariant else None

    def decode(control):
        if enhanced is not None:
            return model.decode_features(enhanced, data['prompt_indices'], control, True)
        return model(data['points'], data['prompt_indices'], control, features, True)

    predictions, energies, middle = [], [], []
    for control in controls.unbind(1):
        pred, aux = decode(control)
        predictions.append(pred)
        if 'energy' in aux:
            energies.append(aux['energy'])
    if midpoints:
        for level in range(controls.shape[1] - 1):
            middle.append(decode((controls[:, level] + controls[:, level + 1]) / 2)[0])
    return (torch.stack(predictions, 1), torch.stack(energies, 1) if energies else None,
            torch.stack(middle, 1) if middle else None)


def clean_env():
    env = dict(os.environ)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


class DDPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_no_environment_and_eval_do_not_initialize(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(dist, 'init_process_group') as init:
            for device in ('cpu', 'cuda:0'):
                ctx = init_distributed('train', device)
                self.assertFalse(ctx.distributed)
                self.assertEqual((ctx.rank, ctx.local_rank, ctx.world_size), (0, 0, 1))
                self.assertEqual(str(ctx.device), device)
            init.assert_not_called()
        with patch.dict(os.environ, {'RANK': '2', 'LOCAL_RANK': '2', 'WORLD_SIZE': '4'}, clear=True):
            with patch.object(dist, 'init_process_group') as init:
                for action in ('eval', 'export', 'audit-ids'):
                    ctx = init_distributed(action, 'cuda')
                    self.assertFalse(ctx.distributed)
                    self.assertFalse(ctx.is_primary)
                init.assert_not_called()
        with patch.dict(os.environ, {'RANK': '0'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'together'):
                init_distributed('train', 'cpu')

    def test_cuda_torchrun_uses_local_rank_nccl(self):
        with patch.dict(os.environ, {'RANK': '2', 'LOCAL_RANK': '2', 'WORLD_SIZE': '4'}, clear=True):
            with patch.object(torch.cuda, 'set_device') as select, patch.object(dist, 'init_process_group') as init:
                ctx = init_distributed('train', 'cuda:0')
                select.assert_called_once_with(2)
                self.assertEqual(init.call_args.kwargs['backend'], 'nccl')
                self.assertEqual(str(ctx.device), 'cuda:2')
                self.assertTrue(ctx.distributed)

    def test_sampler_object_disjoint_equal_batches_and_epoch(self):
        for ids in ([f'obj-{i}' for i in range(19)],
                    [f'obj-{i}' for i in range(19) for _ in range(i % 5 + 1)]):
            samplers = [DistributedObjectBalancedSampler(ids, 4, rank, batch_size=2, seed=17)
                        for rank in range(4)]
            shards = [list(s) for s in samplers]
            self.assertEqual([len(s) for s in samplers], [4] * 4)
            flat_ids = [ids[i] for shard in shards for i in shard]
            self.assertEqual(len(set(flat_ids)), 16)
            self.assertEqual(samplers[0].dropped_objects, 3)
            self.assertEqual(shards, [list(s) for s in samplers])
            for s in samplers:
                s.set_epoch(1)
            self.assertNotEqual(shards, [list(s) for s in samplers])
            self.assertEqual(len({ids[i] for s in samplers for i in s}), 16)
        # Unequal chain counts do not change each object's sampling frequency.
        ids = ['many'] * 20 + ['one']
        sampler = DistributedObjectBalancedSampler(ids, seed=4)
        seen = set()
        for epoch in range(20):
            sampler.set_epoch(epoch)
            indices = list(sampler)
            self.assertEqual(sorted(ids[i] for i in indices), ['many', 'one'])
            seen.update(i for i in indices if ids[i] == 'many')
        self.assertGreater(len(seen), 5)
        with self.assertRaisesRegex(ValueError, 'global batch'):
            DistributedObjectBalancedSampler(['a', 'b'], 4, 0)

    def test_precision_and_fp32_geometry_loss(self):
        self.assertEqual(resolve_precision(amp=True, device='cuda'), 'fp16')
        self.assertEqual(resolve_precision(amp=True, device='cpu'), 'bf16')
        self.assertEqual(resolve_precision('none', amp=True), 'none')
        self.assertEqual(resolve_precision(), 'none')
        self.assertEqual(autocast_dtype('bf16'), torch.bfloat16)
        self.assertEqual(autocast_dtype('fp16'), torch.float16)
        self.assertIsNone(autocast_dtype('none'))
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else [])
        for device in devices:
            for precision in ('bf16', 'none'):
                with self.subTest(device=device, precision=precision):
                    model = HyperSegH(**model_options()).to(device)
                    trainer = HyperSegTrainer(model, torch.optim.AdamW(model.parameters()), device, precision=precision)
                    self.assertIsNone(trainer.scaler)
                    data = {k: v.to(device) for k, v in batch().items()}
                    with torch.autocast(device, enabled=precision != 'none', dtype=torch.bfloat16):
                        pred, energy, middle = trainer.predict_levels(data)
                    self.assertEqual(energy.dtype, torch.float32)
                    self.assertEqual(pred.dtype, torch.bfloat16 if precision == 'bf16' else torch.float32)
                    total, parts = trainer.objective(pred, data['labels'], data['valid'], energy, middle)
                    self.assertEqual(total.dtype, torch.float32)
                    self.assertTrue(all(v.dtype == torch.float32 for v in parts.values()))
                    self.assertTrue(all(np.isfinite(v) for v in trainer.step(data).values()))

    def test_hierarchy_forward_matches_old_algorithm_all_variants(self):
        data = batch()
        for variant in ('legacy', 'radius-film', 'hyperseg-h', 'spherical-hierarchy'):
            for track in ('A1', 'A2'):
                with self.subTest(variant=variant, track=track):
                    model = HyperSegH(**model_options(model_variant=variant)).eval()
                    if track == 'A1':
                        model.control_signal = 'scale' if variant == 'legacy' else 'scale-proxy'
                    trainer = HyperSegTrainer(model, torch.optim.AdamW(model.parameters()), 'cpu', track=track)
                    expected = reference_predict(model, data, midpoints=track != 'A1')
                    with patch.object(model.enhancer, 'forward', wraps=model.enhancer.forward) as encode:
                        actual = trainer.predict_levels(data)
                        expected_calls = 1 if model.encoder_is_control_invariant else (5 if track == 'A2' else 3)
                        self.assertEqual(encode.call_count, expected_calls)
                    for got, want in zip(actual, expected):
                        if want is None:
                            self.assertIsNone(got)
                        else:
                            torch.testing.assert_close(got, want, rtol=0, atol=0)
                    trainer.step(data)
                    self.assertEqual([name for name, p in model.named_parameters()
                                      if p.requires_grad and p.grad is None], [])

    def test_raw_backbone_once_and_frozen_eval(self):
        backbone = torch.nn.Sequential(torch.nn.Linear(3, 12), torch.nn.Dropout(.5))
        model = HyperSegH(**model_options(), backbone=backbone, freeze_backbone=True)
        trainer = HyperSegTrainer(model, torch.optim.AdamW(model.parameters()), 'cpu')
        data = batch()
        del data['features']
        data['backbone_points'] = data['points'] * .7
        with patch.object(backbone, 'forward', wraps=backbone.forward) as encode:
            with patch.object(model.enhancer, 'forward', wraps=model.enhancer.forward) as enhance:
                trainer.step(data)
                self.assertEqual(encode.call_count, 1)
                self.assertEqual(enhance.call_count, 1)
                torch.testing.assert_close(encode.call_args.args[0], data['backbone_points'])
        self.assertFalse(backbone.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in backbone.parameters()))

    def test_official_backbone_inactive_heads_remain_in_checkpoint(self):
        from hyperseg_h.backbone import OfficialBackbone

        class Architecture(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.pvcnn = torch.nn.Linear(3, 12)
                self.triplane_transformer = torch.nn.Linear(12, 12)
                self.sdf_decoder = torch.nn.Linear(12, 1)
                self.logit_scale = torch.nn.Parameter(torch.ones(1))

        modules = {name: ModuleType(name) for name in (
            'partfield.config.defaults', 'partfield.model_trainer_pvcnn_only_demo',
            'partfield.model.PVCNN.encoder_pc')}
        modules['partfield.config.defaults']._C = SimpleNamespace(
            clone=lambda: SimpleNamespace(merge_from_file=lambda path: None))
        modules['partfield.model_trainer_pvcnn_only_demo'].Model = Architecture
        modules['partfield.model.PVCNN.encoder_pc'].sample_triplane_feat = lambda planes, points: planes
        with patch.dict(sys.modules, modules):
            backbone = OfficialBackbone('unused-in-mocked-architecture', load_checkpoint=False)
        self.assertTrue(all(p.requires_grad for p in backbone.model.pvcnn.parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.model.triplane_transformer.parameters()))
        self.assertTrue(all(not p.requires_grad for p in backbone.model.sdf_decoder.parameters()))
        self.assertFalse(backbone.model.logit_scale.requires_grad)
        self.assertEqual(set(backbone.model.state_dict()), set(Architecture(None).state_dict()))

    def test_nonprimary_actions_do_not_write(self):
        from hyperseg_h.cli import main
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'must-not-exist'
            with patch.dict(os.environ, {'RANK': '1', 'LOCAL_RANK': '1', 'WORLD_SIZE': '4'}, clear=True):
                for action in ('eval', 'export', 'audit-ids'):
                    main([action, '--device', 'cpu', '--output', str(output)])
            self.assertFalse(output.exists())

    def test_two_process_gloo_sync_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=2', str(Path(__file__).resolve()), '--worker', temp]
            result = subprocess.run(command, cwd=ROOT, env=clean_env(), capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((Path(temp) / 'verified.json').exists())
            self.assertFalse((Path(temp) / 'forbidden-rank1.pt').exists())

    def test_four_cpu_rank_cli_train_resume_eval_export(self):
        """Four CPU ranks exercise the launch topology; no CUDA/A800 claim."""
        import yaml
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for i in range(6):
                masks = {'root': torch.ones(24, dtype=torch.uint8),
                         'leaf': (torch.arange(24) < 12).byte()}
                cache = {'model_id': f'obj-{i}', 'points': torch.randn(24, 3),
                         'features': torch.randn(24, 12), 'node_masks': masks,
                         'chains': [{'node_ids': ['root', 'leaf']}]}
                torch.save(cache, root / f'{i}.pt')
                records.append({'model_id': f'obj-{i}', 'cache_path': f'{i}.pt'})
            (root / 'train.json').write_text(json.dumps(records[:5]))
            (root / 'test.json').write_text(json.dumps(records[5:]))
            prefix = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                      '--nproc-per-node=4', str(ROOT / 'scripts/hyperseg_h.py')]

            def run(arguments, distributed=True):
                command = prefix if distributed else [sys.executable, str(ROOT / 'scripts/hyperseg_h.py')]
                result = subprocess.run(command + arguments, cwd=ROOT, env=clean_env(),
                                        capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            for track in ('A1', 'A2'):
                options = model_options()
                options.update(control_signal='scale-proxy' if track == 'A1' else 'hierarchy',
                               hierarchy_enabled=track != 'A1',
                               compatibility_mode='official-compatibility')
                config = {'model': options, 'precision': 'bf16',
                          'loss': {'seg_loss_mode': 'official_adaptive' if track == 'A1' else 'per_level_adaptive'}}
                path = root / f'{track}.yaml'
                path.write_text(yaml.safe_dump(config))
                output = root / track
                train_args = ['train', '--device', 'cpu', '--track', track,
                              '--manifest', str(root / 'train.json'),
                              '--test-manifest', str(root / 'test.json'),
                              '--output', str(output), '--num-points', '12', '--batch-size', '1']
                logs = run(train_args + ['--config', str(path)])
                self.assertEqual(logs.count('NON-BENCHMARK RUN'), 1)
                checkpoint = output / 'latest.pt'
                state = torch.load(checkpoint, weights_only=True)
                self.assertEqual(state['epoch'], 1)
                self.assertIsNone(state['scaler'])
                self.assertEqual(state['config']['precision'], 'bf16')
                self.assertFalse(any(k.startswith('module.') for k in state['model_state_dict']))
                provenance = json.loads((output / 'run_config.json').read_text())
                self.assertEqual(provenance['distributed']['global_batch_size'], 4)
                self.assertEqual(provenance['distributed']['objects_dropped_per_epoch'], 1)
                # Load the DDP-produced unprefixed checkpoint in a single process,
                # then resume that single-process checkpoint under torchrun.
                single = root / f'{track}-single'
                run([*train_args[:train_args.index('--output')], '--output', str(single),
                     '--num-points', '12', '--batch-size', '1', '--checkpoint', str(checkpoint),
                     '--resume', '--epochs', '2'], distributed=False)
                single_checkpoint = single / 'latest.pt'
                before = torch.load(single_checkpoint, weights_only=True)
                run(train_args + ['--checkpoint', str(single_checkpoint), '--resume', '--epochs', '3'])
                after = torch.load(checkpoint, weights_only=True)
                self.assertEqual(after['epoch'], 3)
                for key, value in before['optimizer']['state'].items():
                    self.assertEqual(after['optimizer']['state'][key]['step'], value['step'] + 1)
                history = json.loads((output / 'train_metrics.json').read_text())
                self.assertEqual([row['epoch'] for row in history], [0, 2])
                eval_dir = root / f'{track}-eval'
                logs = run(['eval', '--track', track, '--device', 'cpu', '--checkpoint', str(checkpoint),
                            '--manifest', str(root / 'test.json'), '--output', str(eval_dir), '--num-points', '12'])
                self.assertEqual(logs.count('official_interactive_iou_percent'), 1)
                self.assertTrue((eval_dir / 'metrics.json').exists())
                eval_provenance = json.loads((eval_dir / 'run_config.json').read_text())
                self.assertFalse(eval_provenance['distributed']['enabled'])
                self.assertEqual(eval_provenance['distributed']['world_size'], 1)
                exported = root / f'{track}-export.pt'
                logs = run(['export', '--track', track, '--device', 'cpu', '--checkpoint', str(checkpoint),
                            '--output', str(exported)])
                self.assertEqual(logs.count('Exported single checkpoint'), 1)
                self.assertTrue(exported.exists())


def distributed_worker(output):
    """Real Gloo reducer regression with unequal hierarchy depths across ranks."""
    ctx = init_distributed('train', 'cpu')
    try:
        for variant in ('hyperseg-h', 'radius-film', 'legacy', 'spherical-hierarchy'):
            torch.manual_seed(9)
            raw = HyperSegH(**model_options(model_variant=variant))
            reference = copy.deepcopy(raw)
            model = wrap_model(raw, ctx)
            assert not model.find_unused_parameters
            assert unwrap_model(model) is raw
            trainer = HyperSegTrainer(model, torch.optim.AdamW(model.parameters(), lr=.001), 'cpu')
            ref_trainer = HyperSegTrainer(reference, torch.optim.AdamW(reference.parameters(), lr=.001), 'cpu')
            # Two iterations detect missing reduction hooks/unused parameters.
            for step in range(2):
                torch.manual_seed(40 + 10 * ctx.rank + step)
                data = batch(levels=2 + ctx.rank)
                reference.train()
                ref_trainer.optimizer.zero_grad(set_to_none=True)
                pred, energy, middle = reference_predict(reference, data)
                total, _ = ref_trainer.objective(pred, data['labels'], data['valid'], energy, middle)
                total.backward()
                for p in reference.parameters():
                    if p.requires_grad:
                        assert p.grad is not None
                        dist.all_reduce(p.grad)
                        p.grad /= ctx.world_size
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                ref_trainer.optimizer.step()
                with patch.object(model, 'forward', wraps=model.forward) as forward:
                    trainer.step(data)
                    assert forward.call_count == 1
                for expected, actual in zip(reference.parameters(), raw.parameters()):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            flat = torch.cat([p.detach().flatten() for p in raw.parameters()])
            others = [torch.empty_like(flat) for _ in range(ctx.world_size)]
            dist.all_gather(others, flat)
            for other in others:
                torch.testing.assert_close(flat, other, rtol=0, atol=0)
            save_checkpoint(Path(output) / f'{variant}.pt', model, {'model': model_options(model_variant=variant)},
                            optimizer=trainer.optimizer.state_dict(), epoch=2)
            if not ctx.is_primary:
                save_checkpoint(Path(output) / 'forbidden-rank1.pt', model, {})
            dist.barrier()
            stored = torch.load(Path(output) / f'{variant}.pt', weights_only=True)
            assert not any(key.startswith('module.') for key in stored['model_state_dict'])
            load_checkpoint(Path(output) / f'{variant}.pt', model)
            single = HyperSegH(**model_options(model_variant=variant))
            load_checkpoint(Path(output) / f'{variant}.pt', single)
            for expected, actual in zip(raw.parameters(), single.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if ctx.is_primary:
            (Path(output) / 'verified.json').write_text(json.dumps({'backend': 'gloo', 'world_size': ctx.world_size}))
        dist.barrier()
    finally:
        cleanup_distributed(ctx)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        distributed_worker(Path(sys.argv[2]))
    else:
        unittest.main(verbosity=2)

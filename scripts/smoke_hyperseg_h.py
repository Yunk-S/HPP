#!/usr/bin/env python3
"""CPU/CUDA synthetic regression suite. No real benchmark/SOTA claim."""
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from decoder.models.HyperbolicGeometry import HyperbolicGeometry, RadiusController, EntailmentConeGate
from decoder.models.CrossAttentionDecoder import Decoder, CrossAttentionPointsQ
from decoder.models.PointFeatureEnhancer import PointFeatureEnhancer
from hyperseg_h.model import HyperSegH
from hyperseg_h.checkpoint import audit_load, load_checkpoint, load_official, save_checkpoint
from hyperseg_h.data import (unit_sphere, official_encoder_normalize, official_decoder_normalize,
    boundary_prompt, leaf_stratified_sample, validate_cache,
    HierarchyDataset, ObjectBalancedSampler, collate_hierarchy, exclude_overlap)
from hyperseg_h.losses import HierarchyLoss, adaptive_bce_dice, containment_loss, geometric_loss


def options(**kw):
    return dict(feature_dim=12, hidden_dim=24, hyper_dim=8, num_heads=4,
                enhancer_layers=1, decoder_layers=3, dropout=0.,
                compatibility_mode='batch-first-corrected', **kw)


def cache(model_id='object-27', levels=3):
    n = 24
    masks = {'root': torch.ones(n, dtype=torch.uint8),
             'part': torch.cat([torch.ones(12), torch.zeros(12)]).byte(),
             'tiny': torch.cat([torch.ones(2), torch.zeros(22)]).byte()}
    return {'model_id': model_id, 'points': torch.randn(n, 3), 'features': torch.randn(n, 12),
            'node_masks': masks, 'chains': [{'node_ids': ['root', 'part', 'tiny'][:levels]}]}


class SmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_radius_and_geometry(self):
        rc = RadiusController()
        self.assertAlmostEqual(rc.gamma.item(), 1., places=6)
        g = torch.linspace(0, 1, 101, requires_grad=True)
        radius = rc(g)
        self.assertTrue((radius.diff() > 0).all())
        self.assertAlmostEqual(radius[0].item(), .15, places=6)
        self.assertAlmostEqual(radius[-1].item(), .85, places=6)
        radius.sum().backward()
        self.assertTrue(torch.isfinite(g.grad).all() and torch.isfinite(rc.raw_gamma.grad))
        geometry = HyperbolicGeometry()
        for q, k in [(torch.zeros(2, 8), torch.zeros(2, 7, 8)),
                     (torch.randn(2, 8) * 1e4, torch.randn(2, 7, 8) * 1e4)]:
            q.requires_grad_(); k.requires_grad_()
            with torch.autocast('cpu', dtype=torch.bfloat16):
                angle = geometry.cone_angle_closed_form(q, k)
                result = geometry.expmap0(k)
            self.assertEqual(angle.dtype, torch.float32)
            self.assertTrue(torch.isfinite(angle).all())
            self.assertTrue((result.norm(dim=-1) < 1).all())
            (angle.sum() + result.sum()).backward()
            self.assertTrue(torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all())
        q = torch.tensor([[.3, 0.]])
        k = torch.tensor([[[.9, 0.], [-.9, 0.]]])
        angle = geometry.cone_angle_closed_form(q, k)
        self.assertLess(angle[0, 0], .01)
        self.assertGreater(angle[0, 1], 3.)
        self.assertTrue((geometry.aperture(torch.stack([torch.tensor([r, 0.]) for r in [.2,.5,.8]])).diff() < 0).all())
        with self.assertRaises(ValueError):
            rc(torch.tensor([-0.1]))
        with self.assertRaises(ValueError):
            RadiusController(rho_min=0.)

    def test_gate_amp_and_gradient(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            gate = EntailmentConeGate(24, 8).to(device)
            feats = torch.randn(2, 17, 24, device=device, requires_grad=True)
            with torch.autocast(device, dtype=torch.float16 if device == 'cuda' else torch.bfloat16):
                aux = gate(feats, torch.tensor([0, 6], device=device), torch.tensor([0., 1.], device=device))
            self.assertEqual(aux['gates'].dtype, torch.float32)
            self.assertTrue(((aux['gates'] > 0) & (aux['gates'] <= 1)).all())
            sum(value.sum() for value in aux.values()).backward()
            self.assertTrue(torch.isfinite(feats.grad).all())
            for param in gate.parameters():
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_single_prompt_gate_and_multilayer(self):
        layer = CrossAttentionPointsQ(24, 4).eval()
        x, prompt = torch.randn(2, 17, 24), torch.randn(2, 1, 24)
        regular = layer(x, prompt)
        gate = torch.linspace(0, 1, 17).expand(2, -1)
        torch.testing.assert_close(layer(x, prompt, gate), regular * gate[..., None])
        decoder = Decoder(24, 4, num_layers=3, gate_aware=True).eval()
        expected = x
        for block in decoder.decoder_blocks:
            expected, _ = block(expected, prompt, gate, return_prompt=True)
        torch.testing.assert_close(decoder(x, prompt, gate), expected)
        propagated = Decoder(24, 4, num_layers=3, gate_aware=True,
                             prompt_propagation=True).eval()
        expected, updated_prompt = x, prompt
        for block in propagated.decoder_blocks:
            expected, updated_prompt = block(expected, updated_prompt, gate, return_prompt=True)
        torch.testing.assert_close(propagated(x, prompt, gate), expected)
        self.assertFalse(torch.allclose(updated_prompt, prompt))

    def test_official_compatibility_mode_is_explicit(self):
        enhancer = PointFeatureEnhancer(
            feature_dim=12, feature_proj_dim=24, pos_num_feats=8,
            transformer_hidden_dim=24, transformer_num_heads=4,
            transformer_num_layers=2, dropout=0.,
            compatibility_mode='official-compatibility')
        self.assertFalse(enhancer.transformer_blocks[0].attn.batch_first)
        decoder = Decoder(24, 4, num_layers=3, gate_aware=False)
        with self.assertRaises(ValueError):
            decoder(torch.randn(1, 3, 24), torch.randn(1, 1, 24),
                    gates=torch.ones(1, 3))
        # Numerical parity against upstream is intentionally a separate,
        # pinned-checkout test: scripts/check_official_compatibility.py.

    def test_corrected_batch_independence(self):
        model = HyperSegH(**options()).eval()
        x, f = torch.randn(2, 11, 3), torch.randn(2, 11, 12)
        idx, control = torch.tensor([2, 3]), torch.tensor([.2, .9])
        full = model(x, idx, control, f)
        single = torch.cat([model(x[i:i+1], idx[i:i+1], control[i:i+1], f[i:i+1]) for i in range(2)])
        torch.testing.assert_close(full, single, atol=1e-6, rtol=1e-5)

    def test_control_invariant_encoding_is_reused(self):
        model = HyperSegH(**options()).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = __import__('training.decoder_train.code.train', fromlist=['HyperSegTrainer']).HyperSegTrainer(
            model, optimizer, 'cpu')
        batch = {'points': torch.randn(2, 13, 3), 'features': torch.randn(2, 13, 12),
                 'prompt_indices': torch.tensor([0, 2]),
                 'granularities': torch.tensor([[0., .5, 1.], [0., .5, 1.]]),
                 'scales': torch.tensor([[1., .5, .2], [1., .5, .2]])}
        with patch.object(model.enhancer, 'forward', wraps=model.enhancer.forward) as forward:
            trainer.predict_levels(batch)
            self.assertEqual(forward.call_count, 1)

    def test_internal_normalization_modes(self):
        points = torch.tensor([[0., 0., 0.], [2., 4., 8.], [1., 2., 4.]])
        encoder = official_encoder_normalize(points)
        decoder = official_decoder_normalize(points)
        self.assertLessEqual(float(encoder.abs().max()), .900001)
        torch.testing.assert_close(decoder.mean(0), torch.zeros(3), atol=1e-5, rtol=0)

    def test_variants_and_full_backward(self):
        for variant in ['legacy','hyperseg-h','radius-film','spherical-hierarchy']:
            model = HyperSegH(**options(model_variant=variant, control_signal='hierarchy', hierarchy_enabled=True))
            x, f = torch.randn(2, 17, 3), torch.randn(2, 17, 12)
            p, aux = model(x, torch.tensor([0, 5]), torch.tensor([0., 1.]), f, True)
            p.square().mean().backward()
            self.assertEqual(p.shape, (2,17))
            for param in model.parameters():
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all())
            if aux:
                self.assertAlmostEqual(float(aux['rho'][0]), .15, places=6)
        model = HyperSegH(**options()).eval()
        _, aux = model(x, torch.tensor([0,5]), torch.tensor([0.,1.]), f, True)
        torch.testing.assert_close(aux['rho'], torch.tensor([.85,.15]))
        with self.assertRaises(ValueError):
            HyperSegH(**options(control_signal='hierarchy', hierarchy_enabled=False))

    def test_checkpoint_roundtrip_and_rejection(self):
        model = HyperSegH(**options()).eval()
        legacy = HyperSegH(**options(model_variant='legacy', control_signal='scale')).eval()
        checkpoint = {key: {'module.' + k: v for k,v in mod.state_dict().items()} for key, mod in [
            ('point_feature_enhancer_state_dict',legacy.enhancer),('decoder_state_dict', legacy.decoder),
            ('seg_head_state_dict',legacy.seg_head)]}
        reports = load_official(model, checkpoint)
        self.assertTrue(all(r['ratio'] == 1 for r in reports))
        with self.assertRaises(RuntimeError):
            audit_load(model.decoder, {}, 'broken')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'single.pt'
            save_checkpoint(path, model, {'model': options()})
            other = HyperSegH(**options()).eval()
            load_checkpoint(path, other)
            x, f, idx, control = torch.randn(2,9,3),torch.randn(2,9,12),torch.tensor([0,2]),torch.tensor([.2,.8])
            torch.testing.assert_close(model(x,idx,control,f),other(x,idx,control,f),rtol=0,atol=0)
            state = torch.load(path, weights_only=True)
            del state['model_state_dict']['gate.raw_beta']
            torch.save(state,path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path,other)

    def test_losses_validity_and_balancing(self):
        p = torch.tensor([[[.2,.8],[.6,.4],[.9,.9]]],requires_grad=True)
        valid = torch.tensor([[True,True,False]])
        self.assertAlmostEqual(containment_loss(p,valid).item(),.4,places=6)
        masked = torch.tensor([[[True,True],[False,True],[True,True]]])
        self.assertEqual(containment_loss(p,valid,masked).item(),0.)
        e = torch.tensor([[[.1,.2,.3,.4],[.4,.5,.6,.7]]],requires_grad=True)
        target = torch.tensor([[[1.,0.,0.,0.],[1.,1.,1.,0.]]])
        self.assertAlmostEqual(geometric_loss(e,target,torch.ones(1,2,dtype=torch.bool)).item(),.3,places=6)
        total, _ = HierarchyLoss()(p,torch.ones_like(p),valid,middle=(p[:,:-1]+p[:,1:])/2)
        total.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        all_invalid, _ = HierarchyLoss()(p,torch.ones_like(p),torch.zeros_like(valid))
        self.assertEqual(all_invalid.item(),0.)
        sparse_target = torch.tensor([[[1., 0., 0., 0.]]])
        sparse_prob = torch.full_like(sparse_target, .1)
        bce, dice = adaptive_bce_dice(
            sparse_prob, sparse_target, torch.ones_like(sparse_target, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool))
        self.assertTrue(torch.isfinite(bce) and torch.isfinite(dice))
        self.assertEqual(HierarchyLoss().seg_loss_mode, 'official_adaptive')
        self.assertEqual(HierarchyLoss(seg_loss_mode='per_level_adaptive').seg_loss_mode,
                         'per_level_adaptive')

    def test_data_quota_prompt_and_padding(self):
        item = cache()
        validate_cache(item)
        idx = leaf_stratified_sample(item['node_masks'],['tiny'],12,np.random.default_rng(0))
        self.assertEqual(len(idx.unique()),12)
        self.assertEqual(int(item['node_masks']['tiny'][idx].sum()),2)
        points = unit_sphere(item['points'])
        self.assertLessEqual(points.norm(dim=-1).max().item(),1.000001)
        prompt = boundary_prompt(points,item['node_masks']['tiny'],chunk_size=3)
        self.assertTrue(item['node_masks']['tiny'][prompt])
        with self.assertRaises(ValueError):
            boundary_prompt(points,torch.zeros(24))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'a.pt'; torch.save(item,path)
            dataset = HierarchyDataset([{'model_id':item['model_id'],'cache_path':str(path)}],12)
            a,b = dataset[0],dataset[0]
            torch.testing.assert_close(a['points'],b['points'])
            batch = collate_hierarchy([a,{**a,'labels':a['labels'][:2], 'valid':a['valid'][:2],
                 'granularities':a['granularities'][:2],'scales':a['scales'][:2]}])
            self.assertEqual(batch['valid'].tolist(),[[True,True,True],[True,True,False]])
        broken = cache(); broken['node_masks']['tiny'][-1]=1
        with self.assertRaises(ValueError):
            validate_cache(broken)
        sampler = ObjectBalancedSampler(['id-2','id-2','id-99'])
        torch.testing.assert_close(sampler.weights,torch.tensor([.5,.5,1.],dtype=torch.double))

    def test_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            kept, ids = exclude_overlap([{'model_id':'7'},{'model_id':'103'}],[{'model_id':'7'}],tmp)
            self.assertEqual(ids,['7']); self.assertEqual(kept,[{'model_id':'103'}])
            self.assertEqual((Path(tmp)/'excluded_overlap_ids.txt').read_text(),'7\n')

    def test_training_cpu_amp(self):
        from training.decoder_train.code.train import HyperSegTrainer
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            model = HyperSegH(**options(control_signal='hierarchy',hierarchy_enabled=True)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
            trainer = HyperSegTrainer(model,optimizer,device,amp=True)
            batch = {'points':torch.randn(2,12,3),'features':torch.randn(2,12,12),
                     'prompt_indices':torch.tensor([0,1]),'granularities':torch.tensor([[0.,.5,1.],[0.,1.,0.]]),
                     'scales':torch.tensor([[1.,.5,.2],[1.,.2,0.]]),
                     'labels':torch.randint(0,2,(2,3,12)).float(),'valid':torch.tensor([[True]*3,[True,True,False]])}
            before = model.gate.raw_beta.detach().clone()
            result = trainer.step(batch)
            self.assertTrue(all(np.isfinite(v) for v in result.values()))
            self.assertFalse(torch.equal(before,model.gate.raw_beta))

    def test_cli_tracks_export_and_id_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            for name in ['train','test']:
                torch.save(cache(name),tmp/f'{name}.pt')
                (tmp/f'{name}.json').write_text(json.dumps([{'model_id':name,'cache_path':f'{name}.pt'}]))
            (tmp/'config.json').write_text(json.dumps({'model':options(control_signal='hierarchy',hierarchy_enabled=True,model_variant='hyperseg-h')}))
            def cli(*args, fail=False):
                proc = subprocess.run([sys.executable,str(ROOT/'scripts/hyperseg_h.py'),*map(str,args)],
                    cwd=tmp,capture_output=True,text=True,env={**__import__('os').environ,'OMP_NUM_THREADS':'1'})
                if fail:
                    self.assertNotEqual(proc.returncode,0,proc.stdout)
                else:
                    self.assertEqual(proc.returncode,0,proc.stdout+'\n'+proc.stderr)
                return proc
            common = ['--num-points','12','--device','cpu']
            cli('train','--track','A2','--config',tmp/'config.json','--manifest',tmp/'train.json',
                '--test-manifest',tmp/'test.json','--output',tmp/'run',*common)
            checkpoint=tmp/'run/latest.pt'
            for track in ['A2','C']:
                cli('eval','--track',track,'--checkpoint',checkpoint,'--manifest',tmp/'test.json',
                    '--output',tmp/track,'--sweep-steps','3',*common)
                metrics=json.loads((tmp/track/'metrics.json').read_text())
                self.assertIn('same_point_ancestor_consistency',metrics)
                self.assertEqual(metrics['official_interactive_iou_percent'], metrics['object_mean_iou_percent'])
                if track=='C': self.assertEqual(len(metrics['sweep']),3)
            cli('eval','--track','A2','--checkpoint',checkpoint,'--manifest',tmp/'train.json',
                '--output',tmp/'invalid',*common,fail=True)
            cli('export','--track','A2','--checkpoint',checkpoint,'--output',tmp/'export.pt',*common)
            self.assertTrue((tmp/'export.pt').exists())
            for name in ['train','test']:
                flat=cache(name); flat['chains']=[{'node_ids':['part']}]
                torch.save(flat,tmp/f'{name}.pt')
            cli('train','--track','A1','--config',tmp/'config.json','--control-signal','scale-proxy',
                '--manifest',tmp/'train.json','--test-manifest',tmp/'test.json','--output',tmp/'a1',*common)
            cli('eval','--track','A1','--checkpoint',tmp/'a1/latest.pt','--manifest',tmp/'test.json',
                '--output',tmp/'a1eval',*common)
            cli('eval','--track','C','--checkpoint',tmp/'a1/latest.pt','--manifest',tmp/'test.json',
                '--output',tmp/'badproxy',*common,fail=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)

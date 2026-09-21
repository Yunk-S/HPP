import logging
import os

import torch
from tqdm import tqdm

try:
    from . import loss
except ImportError:
    import loss


class Trainer(object):
    def __init__(
        self,
        rank,
        config,
        PointFeatureEnhancer,
        decoder,
        seg_head,
        optimizer,
        scheduler,
        train_loader,
        device,
    ):
        self.rank = rank
        self.config = config
        self.pointfeatureenhancer = PointFeatureEnhancer
        self.decoder = decoder
        self.seg_head = seg_head
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.epoch = 0
        self.step = 0
        self.device = device
        self.loss = loss.AdaptiveLoss()
        self.enhancefeat_dim = config.enhancer.enhancefeat_dim

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        from hyperseg_h.checkpoint import audit_load
        audit_load(self.pointfeatureenhancer, checkpoint["point_feature_enhancer_state_dict"], "enhancer")
        audit_load(self.decoder, checkpoint["decoder_state_dict"], "decoder")
        audit_load(self.seg_head, checkpoint["seg_head_state_dict"], "seg_head")
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            logging.warning("Failed to load optimizer state: %s", e)
        try:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as e:
            logging.warning("Failed to load scheduler state: %s", e)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        logging.info("Loaded checkpoint from %s (epoch=%s step=%s)", path, self.epoch, self.step)

    def save_model(self, name):
        torch.save(
            {
                "point_feature_enhancer_state_dict": self.pointfeatureenhancer.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "seg_head_state_dict": self.seg_head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
            },
            os.path.join(self.config.ckpt_dir, "{}.pt".format(name)),
        )

    def get_scale_dropout_rate(self, epoch):
        return float(self.config.get("scale_dropout_rate", 0.1))

    def train_one_epoch(self):
        self.pointfeatureenhancer.train()
        self.decoder.train()
        self.seg_head.train()

        current_dropout_rate = self.get_scale_dropout_rate(self.epoch)
        enhancer = getattr(self.pointfeatureenhancer, "module", self.pointfeatureenhancer)
        if hasattr(enhancer, "scale_dropout_rate"):
            enhancer.scale_dropout_rate = current_dropout_rate

        data_iter = tqdm(
            self.train_loader,
            desc="Train epoch {}".format(self.epoch),
            disable=(self.rank != 0),
        )
        for data in data_iter:
            self.step += 1
            self.optimizer.zero_grad()

            point_feat = data["feat"].to(self.device)
            prompt_indices = data["prompt_indices"].to(self.device)
            point_coords = data["coord"].to(self.device)

            points_per_batch = self.config.dataset.num_points
            batch_size = len(prompt_indices)
            point_feat = point_feat.view(batch_size, points_per_batch, -1)
            point_coords = point_coords.view(batch_size, points_per_batch, 3)

            continuous_scales = None
            if self.config.get("use_continuous_scale", True):
                continuous_scales = data.get("continuous_scales", None)
                if continuous_scales is not None:
                    continuous_scales = continuous_scales.to(self.device)

            enhance_feat = self.pointfeatureenhancer(
                point_feat, point_coords, None, continuous_scales
            )
            enhance_feat = enhance_feat.view(
                batch_size * points_per_batch, self.enhancefeat_dim
            )
            prompt_feat = enhance_feat.index_select(0, prompt_indices)
            prompt_feat = prompt_feat.view(batch_size, 1, self.enhancefeat_dim)

            enhance_feat = enhance_feat.view(batch_size, points_per_batch, self.enhancefeat_dim)
            decoder_output = self.decoder(enhance_feat, prompt_feat)
            seg_pred = self.seg_head(decoder_output)

            labels = data["label"].to(self.device).float()
            loss_val = self.loss(seg_pred, labels)
            loss_val.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.rank == 0:
                try:
                    data_iter.set_postfix({"loss": "{:.4f}".format(loss_val.item())})
                except Exception:
                    pass

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch()
            if self.rank == 0:
                self.save_model("latest")
                if self.epoch % self.config.training.save_freq == 0:
                    self.save_model("epoch_{}".format(self.epoch))


class HyperSegTrainer:
    """Single-device hierarchy trainer; legacy distributed Trainer remains available."""
    def __init__(self, model, optimizer, device, track='A2', amp=False, loss_options=None):
        from hyperseg_h.losses import HierarchyLoss
        self.model, self.optimizer, self.device = model, optimizer, torch.device(device)
        self.track, self.amp = track, amp
        self.objective = HierarchyLoss(**(loss_options or {}))
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=amp and self.device.type == 'cuda')

    def predict_levels(self, batch):
        outputs, energies = [], []
        features = batch.get('features')
        if features is None and self.model.backbone is not None:
            features = self.model.backbone(batch['points'])
        controls = batch['granularities'] if self.model.control_signal == 'hierarchy' else batch['scales']
        for level in range(controls.shape[1]):
            pred, aux = self.model(batch['points'], batch['prompt_indices'], controls[:, level], features, True)
            outputs.append(pred)
            if 'energy' in aux:
                energies.append(aux['energy'])
        middle = []
        if self.track != 'A1' and self.objective.weights[-1] > 0:
            for level in range(controls.shape[1] - 1):
                control = (controls[:, level] + controls[:, level + 1]) / 2
                middle.append(self.model(batch['points'], batch['prompt_indices'], control, features))
        return (torch.stack(outputs, 1), torch.stack(energies, 1) if energies else None,
                torch.stack(middle, 1) if middle else None)

    def step(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.amp,
                            dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
            probs, energy, middle = self.predict_levels(batch)
        total, parts = self.objective(probs, batch['labels'], batch['valid'], energy, middle)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite training loss')
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {'loss': float(total.detach()), **{k: float(v.detach()) for k, v in parts.items()}}

import torch
from torch import nn
from torch.nn import functional as F


def _level_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


def containment_loss(probs, valid, point_valid=None):
    """All ordered coarse/fine pairs; validity intersects BOTH levels and points."""
    numerator = probs.sum() * 0
    denominator = probs.sum() * 0
    for i in range(probs.shape[1]):
        for j in range(i + 1, probs.shape[1]):
            pair = (valid[:, i] & valid[:, j])[:, None]
            if point_valid is not None:
                pair = pair & point_valid[:, i] & point_valid[:, j]
            numerator = numerator + (F.relu(probs[:, j] - probs[:, i]) * pair).sum()
            denominator = denominator + (probs[:, j] * pair).sum()
    return numerator / denominator.clamp_min(1e-6)


def geometric_loss(energy, targets, valid, margin=0.2, point_valid=None):
    support = torch.ones_like(targets, dtype=torch.bool) if point_valid is None else point_valid
    positive, negative = (targets > 0.5) & support, (targets <= 0.5) & support
    pos = (energy * positive).sum(-1) / positive.sum(-1).clamp_min(1)
    neg = (F.relu(margin - energy) * negative).sum(-1) / negative.sum(-1).clamp_min(1)
    return _level_mean(pos + neg, valid & support.any(-1))


def sandwich_loss(coarse, middle, fine, valid):
    violation = (F.relu(middle - coarse) + F.relu(fine - middle)).mean(-1)
    return _level_mean(violation, valid)


def adaptive_bce_dice(probs, targets, support, valid, epsilon=1e-6, per_level=False):
    """S²AM3D's foreground-adaptive BCE + Dice objective.

    The official decoder weights positive points by
    ``(1 - foreground_fraction) / (foreground_fraction + epsilon)`` before
    combining BCE and Dice.  ``per_level=False`` is the exact flattened
    official reduction; the per-level variant is the explicit hierarchy
    extension used for A2/C.
    """
    support = support.bool()
    if per_level:
        count = support.sum(-1).clamp_min(1)
        positive_fraction = (targets * support).sum(-1) / count
        beta = (1 - positive_fraction) / (positive_fraction + epsilon)
        weights = torch.where(targets > 0.5, beta[..., None], torch.ones_like(targets))
    else:
        active = support & valid[..., None]
        count = active.sum().clamp_min(1)
        positive_fraction = (targets * active).sum() / count
        beta = (1 - positive_fraction) / (positive_fraction + epsilon)
        weights = torch.where(targets > 0.5, beta, torch.ones_like(targets))
    point_bce = F.binary_cross_entropy(
        probs.clamp(epsilon, 1 - epsilon), targets, weight=weights, reduction='none')
    if per_level:
        bce = _level_mean((point_bce * support).sum(-1) / count, valid)
    else:
        bce = (point_bce * active).sum() / count
    dice_support = support & valid[..., None]
    p, t = probs * dice_support, targets * dice_support
    dice = _level_mean(
        1 - (2 * (p * t).sum(-1) + epsilon) / (p.sum(-1) + t.sum(-1) + epsilon),
        valid)
    return bce, dice


class HierarchyLoss(nn.Module):
    def __init__(self, bce_weight=0.7, dice_weight=0.3, contain_weight=0.1,
                 geo_weight=0.1, control_weight=0.1, margin=0.2,
                 seg_loss_mode='official_adaptive', epsilon=1e-6):
        super().__init__()
        self.weights = (bce_weight, dice_weight, contain_weight, geo_weight, control_weight)
        self.margin = margin
        if seg_loss_mode not in ('official_adaptive', 'per_level_adaptive', 'plain_bce'):
            raise ValueError('Unknown seg_loss_mode')
        self.seg_loss_mode = seg_loss_mode
        self.epsilon = epsilon

    def forward(self, probs, targets, valid, energy=None, middle=None, point_valid=None):
        probs, targets = probs.float(), targets.float()
        support = torch.ones_like(targets, dtype=torch.bool) if point_valid is None else point_valid
        valid = valid & support.any(-1)
        if self.seg_loss_mode in ('official_adaptive', 'per_level_adaptive'):
            bce, dice = adaptive_bce_dice(
                probs, targets, support, valid, self.epsilon,
                per_level=self.seg_loss_mode == 'per_level_adaptive')
        else:
            bce = F.binary_cross_entropy(probs.clamp(self.epsilon, 1 - self.epsilon), targets, reduction='none')
            bce = _level_mean((bce * support).sum(-1) / support.sum(-1).clamp_min(1), valid)
            p, t = probs * support, targets * support
            dice = _level_mean(1 - (2 * (p * t).sum(-1) + self.epsilon) /
                               (p.sum(-1) + t.sum(-1) + self.epsilon), valid)
        contain = containment_loss(probs, valid, point_valid)
        geo = probs.sum() * 0 if energy is None else geometric_loss(energy, targets, valid, self.margin, point_valid)
        control = probs.sum() * 0
        if middle is not None and probs.shape[1] > 1:
            pair_valid = valid[:, :-1] & valid[:, 1:]
            if point_valid is None:
                control = sandwich_loss(probs[:, :-1], middle.float(), probs[:, 1:], pair_valid)
            else:
                pair_points = support[:, :-1] & support[:, 1:]
                terms = F.relu(middle.float() - probs[:, :-1]) + F.relu(probs[:, 1:] - middle.float())
                control = _level_mean((terms * pair_points).sum(-1) / pair_points.sum(-1).clamp_min(1), pair_valid)
        parts = dict(bce=bce, dice=dice, containment=contain, geometry=geo, control=control)
        return sum(w * v for w, v in zip(self.weights, parts.values())), parts

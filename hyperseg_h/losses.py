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


class HierarchyLoss(nn.Module):
    def __init__(self, bce_weight=0.7, dice_weight=0.3, contain_weight=0.1,
                 geo_weight=0.1, control_weight=0.1, margin=0.2):
        super().__init__()
        self.weights = (bce_weight, dice_weight, contain_weight, geo_weight, control_weight)
        self.margin = margin

    def forward(self, probs, targets, valid, energy=None, middle=None, point_valid=None):
        probs, targets = probs.float(), targets.float()
        support = torch.ones_like(targets, dtype=torch.bool) if point_valid is None else point_valid
        valid = valid & support.any(-1)
        bce = F.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), targets, reduction='none')
        bce = _level_mean((bce * support).sum(-1) / support.sum(-1).clamp_min(1), valid)
        p, t = probs * support, targets * support
        dice = _level_mean(1 - (2 * (p * t).sum(-1) + 1e-6) / (p.sum(-1) + t.sum(-1) + 1e-6), valid)
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

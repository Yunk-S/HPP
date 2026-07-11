from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


def radius_regularization(
    outputs: List[Dict[str, torch.Tensor]],
    weight: float = 0.01,
    target_mean: float = 1.0,
    epsilon: float = 1e-6,
    ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Radius regularization to prevent HRA collapse or over-extension.
    
    This regularization penalizes:
    1. Radii that are too small (collapse to origin)
    2. Radii that are too large (unstable hyperbolic geometry)
    
    The loss encourages radius to stay around target_mean, providing a balance
    between the two extremes.
    
    Formula:
        L_reg = weight * mean((radius - target_mean)^2)
    
    Args:
        outputs: List of model outputs, each containing 'radius' tensor [B, N_prompts]
        weight: Weight for the regularization term
        target_mean: Target mean radius value (default 1.0 for Poincaré ball)
        epsilon: Small value to prevent numerical issues
        
    Returns:
        Regularization loss (scalar tensor)
    """
    # 必须与主 loss 同 device/dtype（DeepSpeed bf16 下混入 float32/CPU 会在 backward 报
    # "Found dtype Float but expected BFloat16"）。
    _ref = ref
    if _ref is None:
        for output in outputs:
            r = output.get("radius")
            if r is not None and isinstance(r, torch.Tensor):
                _ref = r
                break
    if _ref is None:
        for output in outputs:
            m = output.get("masks")
            if m is not None and isinstance(m, torch.Tensor):
                _ref = m
                break
    if _ref is None:
        return torch.tensor(0.0)

    device, dtype = _ref.device, _ref.dtype
    total_reg = torch.zeros((), device=device, dtype=dtype)
    count = 0

    for output in outputs:
        radius = output.get("radius")
        if radius is not None and isinstance(radius, torch.Tensor):
            deviation = (radius - target_mean) ** 2
            total_reg = total_reg + deviation.mean()
            count += 1

    if count == 0:
        return torch.zeros((), device=device, dtype=dtype)

    return weight * (total_reg / count)


def radius_diversity_loss(
    outputs: List[Dict[str, torch.Tensor]],
    weight: float = 0.005,
    epsilon: float = 1e-6,
    ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Radius diversity loss to encourage varied granularity control.
    
    This loss encourages different prompts to have different radii,
    preventing the model from using uniform attention sharpness.
    
    Formula:
        L_div = -weight * std(radius)
    
    Args:
        outputs: List of model outputs, each containing 'radius' tensor [B, N_prompts]
        weight: Weight for the diversity term
        epsilon: Small value to prevent numerical issues
        
    Returns:
        Diversity loss (scalar tensor)
    """
    _ref = ref
    if _ref is None:
        for output in outputs:
            r = output.get("radius")
            if r is not None and isinstance(r, torch.Tensor):
                _ref = r
                break
    if _ref is None:
        for output in outputs:
            m = output.get("masks")
            if m is not None and isinstance(m, torch.Tensor):
                _ref = m
                break
    if _ref is None:
        return torch.tensor(0.0)

    device, dtype = _ref.device, _ref.dtype
    total_div = torch.zeros((), device=device, dtype=dtype)
    count = 0

    for output in outputs:
        radius = output.get("radius")
        if radius is not None and isinstance(radius, torch.Tensor):
            std_dev = radius.std()
            total_div = total_div - std_dev
            count += 1

    if count == 0:
        return torch.zeros((), device=device, dtype=dtype)

    return weight * (total_div / count)


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-3,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape, [B, ..., N].
                The (probability) predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        reduction: ``'none'`` | ``'mean'`` | ``'sum'``
        eps: A small epsilon value to avoid division by zero.

    Returns:
        torch.Tensor: If reduction is 'none', then the shape is [B, ...]. Otherwise, a scalar is returned.

    References:
        https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
        https://github.com/UX-Decoder/Semantic-SAM/blob/3d6a43a0f8e77167c0013d14067933a78e2d1f5a/semantic_sam/modules/criterion_interactive_many_to_many.py#L57
        https://github.com/open-mmlab/mmdetection/blob/cfd5d3a985b0249de009b67d04f37263e11cdf3d/mmdet/models/losses/dice_loss.py#L9
    """
    assert inputs.shape == targets.shape, (inputs.shape, targets.shape)
    assert inputs.dtype == targets.dtype, (inputs.dtype, targets.dtype)

    numerator = 2 * (inputs * targets).sum(-1)
    # NOTE: If target is binary, target equals to target.square()
    denominator = inputs.square().sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


def compute_mask_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_weight_dice: float = 2
):
    """Loss for mask prediction.

    Args:
        logits: A float tensor of shape [B, C, N]. Multi-mask predicted logits.
        labels: A float tensor of shape [B, N]. Ground-truth binary masks.

    Returns:
        torch.Tensor: [B, C]. Mask loss
    """
    assert logits.dim() == 3, logits.shape
    _labels = labels.unsqueeze(1).expand_as(logits)
    _labels = _labels.to(dtype=logits.dtype)
    loss_ce = sigmoid_focal_loss(logits, _labels, alpha=-1, reduction="none")
    loss_dice = dice_loss(logits.sigmoid(), _labels, reduction="none")
    loss = loss_ce.mean(-1) + loss_weight_dice * loss_dice
    return loss


def compute_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = None):
    """Compute intersection-over-union (IoU).

    Args:
        logits: A float tensor of shape [..., N]. Multi-mask predicted logits.
        targets: A bool tensor of shape [..., N]. Ground-truth binary masks.
        threshold: A float value for thresholding predictions.
            If None, use the default threshold of 0.5.

    Returns:
        torch.Tensor: [...]. IoU scores
    """
    assert logits.shape == targets.shape, (logits.shape, targets.shape)
    assert targets.dtype == torch.bool, targets.dtype
    if threshold is None:
        preds = logits > 0
    else:
        preds = logits.sigmoid() > threshold
    return (preds & targets).sum(-1) / (preds | targets).sum(-1)


@torch.jit.script
def compute_jaccard(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-3):
    assert logits.shape == targets.shape, (logits.shape, targets.shape)
    probs = logits.sigmoid()
    numerator = (probs * targets).sum(-1)
    denominator = (probs.square() + targets.square()).sum(-1) - numerator
    return (numerator + eps) / (denominator + eps)


class Criterion(nn.Module):
    def __init__(
        self,
        use_soft_iou: bool = False,
        radius_reg_weight: float = 0.01,
        radius_target_mean: float = 1.0,
        radius_diversity_weight: float = 0.005,
    ):
        """
        Loss criterion for HPP-SAM.
        
        Args:
            use_soft_iou: Whether to use soft IoU for IoU head loss
            radius_reg_weight: Weight for radius regularization (prevents collapse)
            radius_target_mean: Target mean radius value
            radius_diversity_weight: Weight for radius diversity loss
        """
        super().__init__()
        self.use_soft_iou = use_soft_iou
        self.radius_reg_weight = radius_reg_weight
        self.radius_target_mean = radius_target_mean
        self.radius_diversity_weight = radius_diversity_weight

    def forward(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        gt_masks,
    ) -> tuple:
        # gt_mask: [B*M, N]
        # Follow the "Making the model ambiguity-aware" in Appendix A of SAM.
        # Multimask is only enabled with more than one prompt.
        losses = []
        aux_outputs = []
        for i, output in enumerate(outputs):
            masks = output["masks"]  # [B*M, C, N]
            iou_preds = output["iou_preds"]  # [B*M, C]

            loss_mask = compute_mask_loss(masks, gt_masks)  # [B*M,C]
            if i == 0:
                loss_mask, min_loss_idx = loss_mask.min(dim=1)  # [B*M]
                batch_idx = torch.arange(min_loss_idx.shape[0])
                best_masks = masks[batch_idx, min_loss_idx]  # [B*M, N]
                iou_preds = iou_preds[batch_idx, min_loss_idx]  # [B*M]
            else:
                best_masks = masks.squeeze(1)
                iou_preds = iou_preds.squeeze(1)
            loss_mask = loss_mask.mean()

            iou = compute_iou(best_masks, gt_masks)  # [B*M]，常为 float32；与 bf16 的 iou_preds 混用会破坏图
            iou = iou.to(dtype=iou_preds.dtype)
            if self.use_soft_iou:
                with torch.no_grad():
                    soft_iou = compute_jaccard(
                        best_masks, gt_masks.to(dtype=best_masks.dtype)
                    )
                loss_iou = F.mse_loss(soft_iou, iou_preds)
            else:
                loss_iou = F.mse_loss(iou, iou_preds)

            losses.append(loss_iou + loss_mask)
            aux_outputs.append(
                dict(
                    iou=iou,
                    best_masks=best_masks,
                    loss_mask=loss_mask,
                    loss_iou=loss_iou,
                )
            )

        # Warmup: linear increase from 0.5 (iter=0) to 1.0 (iter=last).
        # Early iters output multimask where model can "get lucky"; later iters accumulate
        # prompt refinement and matter more for convergence. This prevents the first-iter
        # lucky mask from drowning out genuine learning in later rounds.
        if len(losses) > 1:
            step_weights = torch.linspace(
                0.5,
                1.0,
                len(losses),
                device=losses[0].device,
                dtype=losses[0].dtype,
            )
            step_weights = step_weights / step_weights.sum()
            loss = torch.stack(losses) * step_weights
            loss = loss.sum()
        else:
            loss = torch.stack(losses).mean()

        _ref = outputs[0]["masks"]
        # Apply radius regularization if enabled and outputs contain radius
        reg_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        if self.radius_reg_weight > 0:
            reg_loss = reg_loss + radius_regularization(
                outputs,
                weight=self.radius_reg_weight,
                target_mean=self.radius_target_mean,
                ref=_ref,
            )

        if self.radius_diversity_weight > 0:
            reg_loss = reg_loss + radius_diversity_loss(
                outputs,
                weight=self.radius_diversity_weight,
                ref=_ref,
            )

        total_loss = loss + reg_loss

        # Add regularization info to aux_outputs for logging
        if aux_outputs:
            z = torch.zeros((), device=loss.device, dtype=loss.dtype)
            aux_outputs[0]["reg_loss"] = reg_loss.detach()
            aux_outputs[0]["radius_reg"] = (
                radius_regularization(
                    outputs, weight=1.0, target_mean=self.radius_target_mean, ref=_ref
                ).detach()
                if self.radius_reg_weight > 0
                else z
            )
            aux_outputs[0]["radius_div"] = (
                radius_diversity_loss(outputs, weight=1.0, ref=_ref).detach()
                if self.radius_diversity_weight > 0
                else z
            )
        
        return total_loss, aux_outputs

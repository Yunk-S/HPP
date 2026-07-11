"""
Learning diagnostics: overfitting signals, radius health, optional entropy hooks.
Used by train.py when HAS_LEARNING_DIAGNOSTICS is True.

Enhanced version with:
- Comprehensive radius health monitoring
- Curvature tracking (for learnable curvature)
- Mask distribution analysis
- Prediction similarity detection
- Learning stagnation detection
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cosine


class LearningDiagnostics:
    """
    Track train/val stats and suggest interventions (used with AdaptiveRegularization).
    
    Enhanced with:
    - Comprehensive radius health monitoring
    - Curvature tracking
    - Mask diversity analysis
    - Prediction bias detection
    - Learning stagnation detection
    """

    def __init__(
        self,
        patience: int = 10,
        overfit_threshold: float = 0.3,
        collapse_threshold: float = 0.5,
        entropy_threshold: float = 0.3,
        min_epochs: int = 50,
        # New parameters
        radius_min_threshold: float = 0.01,
        radius_max_threshold: float = 10.0,
        curvature_min: float = 1e-6,
        curvature_max: float = 1.0,
        mask_diversity_threshold: float = 0.1,
        prediction_bias_threshold: float = 0.4,
        loss_stagnation_threshold: float = 0.01,
        loss_stagnation_patience: int = 10,
    ):
        self.patience = patience
        self.overfit_threshold = overfit_threshold
        self.collapse_threshold = collapse_threshold
        self.entropy_threshold = entropy_threshold
        self.min_epochs = min_epochs
        
        # Radius thresholds
        self.radius_min_threshold = radius_min_threshold
        self.radius_max_threshold = radius_max_threshold
        
        # Curvature bounds
        self.curvature_min = curvature_min
        self.curvature_max = curvature_max
        
        # Mask diversity
        self.mask_diversity_threshold = mask_diversity_threshold
        self.prediction_bias_threshold = prediction_bias_threshold
        
        # Loss stagnation
        self.loss_stagnation_threshold = loss_stagnation_threshold
        self.loss_stagnation_patience = loss_stagnation_patience

        self.train_history: deque = deque(maxlen=100)
        self.val_history: deque = deque(maxlen=100)
        self.radius_history: deque = deque(maxlen=100)
        self.entropy_history: deque = deque(maxlen=100)
        self.lr_history: deque = deque(maxlen=100)
        self.curvature_history: deque = deque(maxlen=100)
        self.mask_diversity_history: deque = deque(maxlen=100)

        self.bad_epochs = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_iou = 0.0

        self.overfit_counter = 0
        self.collapse_counter = 0
        self.entropy_counter = 0
        self.radius_issue_counter = 0
        self.curvature_issue_counter = 0
        self.mask_collapse_counter = 0
        self.prediction_bias_counter = 0
        self.loss_stagnation_counter = 0

    def record(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.current_epoch = epoch
        self.train_history.append(dict(train_metrics))
        if val_metrics is not None:
            self.val_history.append(dict(val_metrics))
        if lr is not None:
            self.lr_history.append(lr)

        diagnostics = self.compute_diagnostics()

        if val_metrics is not None:
            val_loss = float(val_metrics.get("loss", float("inf")))
            val_iou = float(val_metrics.get("iou", val_metrics.get("IoU_best_multimask_iter0", 0.0)))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou

        return diagnostics

    def record_radius(
        self, 
        radius: torch.Tensor,
        curvature: Optional[float] = None,
    ) -> Dict[str, float]:
        """Record radius statistics and optionally curvature value."""
        r = radius.detach().float().cpu().numpy()
        stats = {
            "radius_mean": float(np.mean(r)),
            "radius_std": float(np.std(r)),
            "radius_min": float(np.min(r)),
            "radius_max": float(np.max(r)),
            "radius_median": float(np.median(r)),
            "radius_q25": float(np.percentile(r, 25)),
            "radius_q75": float(np.percentile(r, 75)),
        }
        self.radius_history.append(stats)
        
        if curvature is not None:
            self.record_curvature(curvature)
        
        return stats
    
    def record_curvature(self, curvature: float) -> None:
        """Record curvature value for tracking."""
        self.curvature_history.append({
            "curvature": float(curvature),
        })
    
    def record_mask_diversity(self, diversity: float) -> None:
        """Record mask diversity score."""
        self.mask_diversity_history.append({"diversity": diversity})

    def compute_diagnostics(self) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "epoch": self.current_epoch,
            "n_train_records": len(self.train_history),
            "n_val_records": len(self.val_history),
        }

        # Train-val gap
        if len(self.train_history) >= 5 and len(self.val_history) >= 5:
            recent_train_loss = float(
                np.mean([h.get("loss", 0.0) for h in list(self.train_history)[-5:]])
            )
            recent_val_loss = float(
                np.mean([h.get("loss", 0.0) for h in list(self.val_history)[-5:]])
            )
            diagnostics["train_val_loss_gap"] = recent_train_loss - recent_val_loss

        # Radius variance ratio
        if len(self.radius_history) >= 2:
            recent = self.radius_history[-1]
            older = self.radius_history[-5] if len(self.radius_history) >= 5 else self.radius_history[0]
            diagnostics["radius_variance_ratio"] = recent["radius_std"] / (older["radius_std"] + 1e-6)
            diagnostics["radius_mean"] = recent["radius_mean"]
            diagnostics["radius_std"] = recent["radius_std"]

        # Curvature tracking
        if len(self.curvature_history) >= 2:
            diagnostics["curvature"] = self.curvature_history[-1]["curvature"]

        # Entropy ratio
        if len(self.entropy_history) >= 2:
            diagnostics["entropy_ratio"] = self.entropy_history[-1] / (self.entropy_history[-5] + 1e-6)

        # Mask diversity
        if len(self.mask_diversity_history) >= 2:
            diagnostics["mask_diversity"] = self.mask_diversity_history[-1]["diversity"]

        # Loss stagnation
        diagnostics["loss_stagnation"] = self._check_loss_stagnation()

        diagnostics["overfit_risk"] = self._compute_overfit_risk()
        diagnostics["health_score"] = self._compute_health_score()
        
        return diagnostics

    def _compute_overfit_risk(self) -> float:
        risk = 0.0
        n = 0
        
        # Train-val gap risk
        if len(self.train_history) >= 5 and len(self.val_history) >= 5:
            rt = float(np.mean([h.get("loss", 0.0) for h in list(self.train_history)[-5:]]))
            rv = float(np.mean([h.get("loss", 0.0) for h in list(self.val_history)[-5:]]))
            gap = rt - rv
            risk += min(1.0, max(0.0, gap / (self.overfit_threshold + 1e-6)))
            n += 1
        
        # Radius collapse risk
        if len(self.radius_history) >= 5:
            ratio = self.radius_history[-1]["radius_std"] / (
                self.radius_history[-5]["radius_std"] + 1e-6
            )
            risk += max(0.0, 1.0 - ratio / self.collapse_threshold)
            n += 1
        
        # Entropy risk
        if len(self.entropy_history) >= 5:
            er = self.entropy_history[-1] / (self.entropy_history[-5] + 1e-6)
            risk += max(0.0, 1.0 - er / self.entropy_threshold)
            n += 1
        
        return risk / max(n, 1)

    def _check_loss_stagnation(self) -> bool:
        """Check if loss has stagnated (no significant improvement)."""
        if len(self.train_history) < self.loss_stagnation_patience:
            return False
        
        recent_losses = [h.get("loss", float("inf")) for h in list(self.train_history)[-self.loss_stagnation_patience:]]
        if len(recent_losses) < 2:
            return False
        
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        avg_loss = np.mean(recent_losses)
        
        if avg_loss > 0:
            relative_change = loss_change / (avg_loss + 1e-6)
            return relative_change < self.loss_stagnation_threshold
        return False

    def _compute_health_score(self) -> float:
        """Compute overall health score [0, 1], higher is better."""
        score = 1.0
        
        # Penalize for radius issues
        if len(self.radius_history) >= 1:
            r = self.radius_history[-1]
            if r["radius_mean"] < self.radius_min_threshold:
                score -= 0.2
            if r["radius_mean"] > self.radius_max_threshold:
                score -= 0.2
            if r["radius_std"] < self.radius_min_threshold:
                score -= 0.1
        
        # Penalize for curvature issues
        if len(self.curvature_history) >= 1:
            c = self.curvature_history[-1]["curvature"]
            if c < self.curvature_min or c > self.curvature_max:
                score -= 0.2
        
        # Penalize for overfit risk
        score -= self._compute_overfit_risk() * 0.3
        
        # Penalize for loss stagnation
        if self._check_loss_stagnation():
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def should_intervene(self) -> Tuple[bool, str]:
        """Check if intervention is needed. Returns (should_intervene, reason)."""
        if self.current_epoch < self.min_epochs:
            return False, "Too early to intervene"
        
        issues = []
        
        # Check overfit risk
        risk = self._compute_overfit_risk()
        if risk > 0.7:
            self.overfit_counter += 1
            if self.overfit_counter >= 3:
                issues.append(f"High overfitting risk ({risk:.2f})")
        else:
            self.overfit_counter = 0
        
        # Check radius collapse
        if len(self.radius_history) >= 5:
            ratio = self.radius_history[-1]["radius_std"] / (
                self.radius_history[-5]["radius_std"] + 1e-6
            )
            if ratio < self.collapse_threshold:
                self.collapse_counter += 1
                if self.collapse_counter >= 3:
                    issues.append(f"HRA radius collapse (ratio {ratio:.3f})")
            else:
                self.collapse_counter = 0
            
            # Check absolute radius bounds
            r_mean = self.radius_history[-1]["radius_mean"]
            if r_mean < self.radius_min_threshold:
                self.radius_issue_counter += 1
                if self.radius_issue_counter >= 3:
                    issues.append(f"Radius too small (mean={r_mean:.4f})")
            else:
                self.radius_issue_counter = 0
            
            if r_mean > self.radius_max_threshold:
                self.radius_issue_counter += 1
                if self.radius_issue_counter >= 3:
                    issues.append(f"Radius too large (mean={r_mean:.4f})")
        
        # Check curvature bounds
        if len(self.curvature_history) >= 1:
            c = self.curvature_history[-1]["curvature"]
            if c < self.curvature_min or c > self.curvature_max:
                self.curvature_issue_counter += 1
                if self.curvature_issue_counter >= 5:
                    issues.append(f"Curvature out of bounds ({c:.6f})")
            else:
                self.curvature_issue_counter = 0
        
        # Check entropy
        if len(self.entropy_history) >= 5:
            er = self.entropy_history[-1] / (self.entropy_history[-5] + 1e-6)
            if er < self.entropy_threshold:
                self.entropy_counter += 1
                if self.entropy_counter >= 3:
                    issues.append(f"Attention entropy dropped (ratio {er:.3f})")
            else:
                self.entropy_counter = 0
        
        # Check loss stagnation
        if self._check_loss_stagnation():
            self.loss_stagnation_counter += 1
            if self.loss_stagnation_counter >= 3:
                issues.append("Loss stagnation detected")
        else:
            self.loss_stagnation_counter = 0
        
        if issues:
            return True, "; ".join(issues)
        return False, "Training is healthy"

    def get_intervention_suggestions(self) -> Dict[str, Any]:
        """Get specific intervention suggestions based on current issues."""
        suggestions = {
            "should_intervene": False,
            "interventions": [],
            "lr_adjustment": None,
            "health_score": self._compute_health_score(),
        }
        
        should_intervene, reason = self.should_intervene()
        suggestions["should_intervene"] = should_intervene
        suggestions["primary_issue"] = reason
        
        if not should_intervene:
            return suggestions
        
        # Generate specific interventions based on issues
        if "collapse" in reason.lower():
            suggestions["interventions"].extend([
                "Increase radius regularization weight",
                "Check if HRA transformation is degenerating",
                "Verify hyperbolic geometry is properly initialized",
            ])
            suggestions["lr_adjustment"] = 0.5  # Reduce LR
        
        if "overfitting" in reason.lower():
            suggestions["interventions"].extend([
                "Increase weight decay",
                "Add dropout if not present",
                "Consider data augmentation",
            ])
            suggestions["lr_adjustment"] = 0.5
        
        if "stagnation" in reason.lower():
            suggestions["interventions"].extend([
                "Learning rate might be too low - try warmup restart",
                "Check gradient flow in HRA modules",
                "Verify curvature is updating properly",
            ])
            suggestions["lr_adjustment"] = 2.0  # Can increase LR if stagnating
        
        if "curvature" in reason.lower():
            suggestions["interventions"].extend([
                "Check curvature bounds - might need wider range",
                "Verify softplus is working in HyperOps",
            ])
        
        return suggestions

    def reset_counters(self) -> None:
        """Reset all issue counters."""
        self.overfit_counter = 0
        self.collapse_counter = 0
        self.entropy_counter = 0
        self.radius_issue_counter = 0
        self.curvature_issue_counter = 0
        self.mask_collapse_counter = 0
        self.prediction_bias_counter = 0
        self.loss_stagnation_counter = 0
        self.bad_epochs = 0


def compute_mask_diversity(pred_masks: torch.Tensor) -> float:
    """
    Compute mask diversity score.
    
    Higher score indicates more diverse predictions across samples.
    Score in range [0, 1].
    """
    if pred_masks.dtype != torch.bool:
        pred_masks = pred_masks > 0
    
    # Point-wise variance (higher variance = more diverse predictions)
    point_variance = pred_masks.float().var(dim=0).mean()
    
    # Unique masks ratio
    unique_masks = torch.unique(pred_masks, dim=0)
    uniqueness_ratio = len(unique_masks) / max(pred_masks.shape[0], 1)
    
    # Combined score
    return float(0.5 * point_variance + 0.5 * min(uniqueness_ratio, 1.0))


def compute_mask_similarity(pred_masks: torch.Tensor) -> float:
    """
    Compute average pairwise cosine similarity between predictions.
    
    Lower score indicates more diverse predictions.
    Score in range [0, 1].
    """
    if pred_masks.dtype != torch.bool:
        pred_masks = pred_masks > 0
    
    # Flatten masks
    masks_flat = pred_masks.float().reshape(pred_masks.shape[0], -1)
    
    # Compute mean cosine similarity
    if masks_flat.shape[0] < 2:
        return 1.0
    
    total_sim = 0.0
    count = 0
    for i in range(masks_flat.shape[0]):
        for j in range(i + 1, masks_flat.shape[0]):
            sim = 1 - cosine(masks_flat[i].numpy(), masks_flat[j].numpy())
            total_sim += sim
            count += 1
    
    return total_sim / max(count, 1)


def check_learning_vs_memorization(
    model_outputs: List[Dict[str, torch.Tensor]],
    gt_masks: torch.Tensor,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Enhanced check for learning vs memorization.
    
    Detects:
    1. IoU not improving across iterations
    2. Low prediction variance (all predictions similar)
    3. Mask collapse (predictions stuck)
    4. Perfect prediction (potential overfitting to training data)
    """
    results: Dict[str, Any] = {
        "is_learning": True, 
        "issues": [],
        "warnings": [],
        "scores": {}
    }
    
    if len(model_outputs) >= 2:
        # Check IoU improvement
        i0 = model_outputs[0]["iou_preds"].mean().item()
        i1 = model_outputs[-1]["iou_preds"].mean().item()
        results["scores"]["iou_improvement"] = i1 - i0
        
        if i1 <= i0:
            results["issues"].append(f"IoU not improving (iter0 {i0:.3f} vs last {i1:.3f})")
            results["is_learning"] = False
    
    # Check prediction variance
    first_masks = model_outputs[0]["masks"][:, 0, :]
    pred_variance = first_masks.std().item()
    results["scores"]["pred_variance"] = pred_variance
    
    if pred_variance < threshold:
        results["issues"].append(f"Low prediction variance ({pred_variance:.4f} < {threshold})")
        results["is_learning"] = False
    
    # Check for mask collapse
    mask_diversity = compute_mask_diversity(first_masks)
    results["scores"]["mask_diversity"] = mask_diversity
    
    if mask_diversity < 0.1:
        results["issues"].append(f"Mask diversity very low ({mask_diversity:.4f})")
        results["is_learning"] = False
    
    # Check for potential overfitting (too high IoU)
    if len(model_outputs) >= 1:
        avg_iou = model_outputs[0]["iou_preds"].mean().item()
        results["scores"]["avg_iou"] = avg_iou
        
        # Very high IoU early in training could indicate memorization
        if avg_iou > 0.95 and model_outputs[0].get("prompt_coords", torch.tensor([])).shape[1] <= 2:
            results["warnings"].append(f"Very high IoU ({avg_iou:.3f}) with few prompts - possible memorization")
    
    # Check prediction bias
    fg_ratio = (first_masks > 0).float().mean().item()
    results["scores"]["fg_ratio"] = fg_ratio
    
    if abs(fg_ratio - 0.5) > 0.45:
        results["warnings"].append(f"Heavy prediction bias (fg_ratio={fg_ratio:.3f})")
    
    return results


def comprehensive_learning_check(
    metrics: Dict[str, float],
    outputs: Optional[List[Dict[str, torch.Tensor]]] = None,
    gt_masks: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Comprehensive learning health check combining multiple signals.
    
    Args:
        metrics: Dictionary of training metrics
        outputs: Optional model outputs for detailed analysis
        gt_masks: Optional ground truth masks
    
    Returns:
        Dictionary with health status and recommendations
    """
    health = {
        "overall_healthy": True,
        "concerns": [],
        "recommendations": [],
        "scores": {},
    }
    
    # Check loss
    loss = metrics.get("loss", float("inf"))
    if np.isnan(loss) or np.isinf(loss):
        health["overall_healthy"] = False
        health["concerns"].append("Loss is NaN or Inf")
        health["recommendations"].append("Check for numerical instability in hyperbolic operations")
    
    # Check IoU trends
    for prefix in ["iou(0)", "iou(1)", "iou_best"]:
        if prefix in metrics:
            iou = metrics[prefix]
            if iou < 0.01:
                health["concerns"].append(f"{prefix} very low ({iou:.4f})")
                health["recommendations"].append("Verify model architecture and data pipeline")
    
    # Check radius health
    for key in ["hra/radius_mean_iter0", "radius_mean"]:
        if key in metrics:
            r_mean = metrics[key]
            r_std = metrics.get(key.replace("mean", "std"), 0.0)
            
            if r_mean < 0.01:
                health["concerns"].append(f"Radius too small ({r_mean:.4f})")
                health["recommendations"].append("Check HRA initialization and curvature settings")
            
            if r_std < 0.001:
                health["concerns"].append(f"Radius collapsed (std={r_std:.6f})")
                health["recommendations"].append("Enable or increase radius regularization")
    
    # Detailed output analysis
    if outputs is not None and gt_masks is not None:
        detailed_check = check_learning_vs_memorization(outputs, gt_masks)
        health["detailed_check"] = detailed_check
        
        if not detailed_check["is_learning"]:
            health["overall_healthy"] = False
            health["concerns"].extend(detailed_check["issues"])
    
    return health

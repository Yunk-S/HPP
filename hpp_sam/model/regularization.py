"""
Adaptive regularization hooks (temp noise, LR hints).
Interventions are conservative; train.py applies LR changes when should_reduce_lr().
"""

from __future__ import annotations

from typing import Any, Dict

import torch


class AdaptiveRegularization:
    def __init__(
        self,
        model: torch.nn.Module,
        base_dropout_p: float = 0.0,
        max_dropout_p: float = 0.3,
        intervention_strength: float = 1.0,
    ):
        self.model = model
        self.base_dropout_p = base_dropout_p
        self.max_dropout_p = max_dropout_p
        self.intervention_strength = intervention_strength
        self.is_intervening = False
        self.intervention_count = 0
        self.intervention_history: list = []

    def intervene(self) -> Dict[str, Any]:
        self.is_intervening = True
        self.intervention_count += 1
        interventions: Dict[str, Any] = {
            "intervention_id": self.intervention_count,
            "actions": [],
        }
        if hasattr(self.model, "hyper_prompt_branch") and hasattr(
            self.model.hyper_prompt_branch, "temp_a"
        ):
            with torch.no_grad():
                noise = torch.randn_like(self.model.hyper_prompt_branch.temp_a) * 0.1
                self.model.hyper_prompt_branch.temp_a.add_(noise)
            interventions["actions"].append("Temperature perturbation (temp_a)")
        interventions["actions"].append("Gradient clip hint (see get_gradient_clip_value)")
        self.intervention_history.append(interventions)
        return interventions

    def should_reduce_lr(self) -> bool:
        return self.intervention_count >= 3

    def get_adjusted_lr(self, base_lr: float) -> float:
        if self.should_reduce_lr():
            return base_lr * (0.5 ** max(self.intervention_count - 2, 1))
        return base_lr

    def get_gradient_clip_value(self, base_clip: float) -> float:
        if self.is_intervening:
            return base_clip * 0.5
        return base_clip


def create_adaptive_regularizer(model: torch.nn.Module) -> AdaptiveRegularization:
    return AdaptiveRegularization(model=model)

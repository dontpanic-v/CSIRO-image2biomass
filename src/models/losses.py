"""Combined regression + auxiliary classification loss for the DINOv3 regressor."""
import torch
import torch.nn as nn

from src.config import ALL_TARGETS, R2_WEIGHTS


class WeightedBiomassLoss(nn.Module):
    """SmoothL1 regression (weighted per target by the competition's R2 weights) plus
    an auxiliary cross-entropy interval-classification loss per target.

    Adapted from the CSIRO-biomass 1st place solution; their fixed loss-weights
    dict is replaced with this project's existing R2_WEIGHTS so there's one
    source of truth for target importance.
    """

    def __init__(self, cls_weight=0.3):
        super().__init__()
        self.regression_loss = nn.SmoothL1Loss(beta=5.0, reduction='none')
        self.classification_loss = nn.CrossEntropyLoss()
        self.cls_weight = cls_weight

    def forward(self, predictions, targets, cls_logits, cls_labels):
        weights = torch.as_tensor(R2_WEIGHTS, device=predictions.device)
        regression_loss = (self.regression_loss(predictions, targets) * weights).mean()

        classification_loss = sum(
            self.classification_loss(cls_logits[target], cls_labels[target])
            for target in ALL_TARGETS
        ) / len(ALL_TARGETS)

        return regression_loss + self.cls_weight * classification_loss

"""DINOv3-based dual-half regressor: shared backbone, cross-view fusion, five regression + classification heads."""
import torch
import torch.nn as nn
from transformers import AutoModel as HFAutoModel

from src.config import ALL_TARGETS
from src.models.attention_fusion import CrossViewAttention


class DINOv3Regressor(nn.Module):
    """DINOv3 backbone shared across both image halves, fused via CrossViewAttention.

    Five independent regression heads have no architectural constraint forcing
    mass balance — that's checked post-hoc in train.py via enforce_mass_balance.
    Matching classification heads predict quantile bins, discarded at inference.
    """

    def __init__(self, model_name, borders, local_only=False, freeze_backbone=False, num_attention_heads=8):
        super().__init__()
        self.backbone = HFAutoModel.from_pretrained(
            model_name, local_files_only=local_only, trust_remote_code=True,
        )
        hidden_size = getattr(self.backbone.config, 'hidden_size', 768)
        self.hidden_dim = hidden_size
        self.borders = borders

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.fusion = CrossViewAttention(hidden_size, num_heads=num_attention_heads)

        def _make_head(out_features):
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, out_features),
            )

        self.regression_heads = nn.ModuleDict({
            target: _make_head(1) for target in ALL_TARGETS
        })
        self.classification_heads = nn.ModuleDict({
            target: _make_head(len(borders[target]) + 1) for target in ALL_TARGETS
        })
        self.softplus = nn.Softplus()

    def _pool(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def forward(self, left, right):
        features_left = self._pool(left)
        features_right = self._pool(right)
        fused = self.fusion(features_left, features_right)

        predictions = torch.cat([
            self.softplus(self.regression_heads[target](fused)) for target in ALL_TARGETS
        ], dim=1)
        cls_logits = {
            target: self.classification_heads[target](fused) for target in ALL_TARGETS
        }
        return predictions, cls_logits

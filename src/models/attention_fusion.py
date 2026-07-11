"""Cross-view self-attention fusion for the DINOv3 regressor's left/right image halves."""
import torch
import torch.nn as nn


class CrossViewAttention(nn.Module):
    """Fuses left/right pooled features via one multi-head self-attention layer + MLP.

    Stacks the two halves into a 2-token sequence so each half attends to the
    other, replacing FiLM conditioning — reimplements the 1st place
    solution's "single MHSA layer + MLP" left/right fusion.
    """

    def __init__(self, dimension, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.ReLU(inplace=True),
            nn.Linear(dimension, dimension),
        )

    def forward(self, features_left, features_right):
        sequence = torch.stack([features_left, features_right], dim=1)  # (B, 2, D)
        attended, _ = self.attention(sequence, sequence, sequence)
        attended = self.norm(attended + sequence)
        return self.mlp(attended.flatten(start_dim=1))  # (B, 2*D) -> (B, D)

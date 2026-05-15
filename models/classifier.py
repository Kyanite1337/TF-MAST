"""Classification head for Stage 3 fine-tuning."""

import torch.nn as nn


class Classifier(nn.Module):
    """Simple MLP on top of the frozen or trainable encoder."""

    def __init__(self, in_dim: int = 256, num_classes: int = 53, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

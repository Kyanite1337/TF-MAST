"""Shared Encoder: CNN stem + Transformer + global pooling.

Input:  (B, C, W)  e.g., (B, 16, 40)
Output: (B, dim)  e.g., (B, 256)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Learnable 1D positional encoding."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))

    def forward(self, x):
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1), :]


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x


class Encoder(nn.Module):
    """Shared encoder: 3 Conv1D layers → position encoding → Transformer → GAP.

    Args:
        in_channels:   input channels (16 for DB5)
        conv_channels: list [in, c1, c2, c3]
        dim:           transformer hidden dim
        num_layers:    number of transformer blocks
        heads:         attention heads
        ffn_dim:       feed-forward hidden dim
        dropout:       dropout rate
    """

    def __init__(
        self,
        in_channels: int = 16,
        conv_channels: list = None,
        dim: int = 256,
        num_layers: int = 6,
        heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [in_channels, 128, 256, 256]

        # CNN stem
        c0, c1, c2, c3 = conv_channels
        self.conv1 = nn.Conv1d(c0, c1, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(c3)

        # Transformer
        self.pos_enc = PositionalEncoding(dim, max_len=512)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(dim)

        self.dim = dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, W)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))  # (B, 256, W//4)

        x = x.transpose(1, 2)  # (B, L, D)
        x = self.pos_enc(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_final(x)
        x = x.mean(dim=1)  # global average pool → (B, D)
        return x

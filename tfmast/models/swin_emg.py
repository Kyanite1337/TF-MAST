from __future__ import annotations

import torch
from torch import nn


class SwinEMGEncoder(nn.Module):
    """Lightweight EMG-grid Transformer backbone with a Swin-like patch interface."""

    def __init__(
        self,
        in_channels: int = 16,
        window_len: int = 40,
        embed_dim: int = 128,
        depths: list[int] | tuple[int, ...] = (2, 2, 2),
        num_heads: list[int] | tuple[int, ...] = (4, 4, 8),
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        patch_size: tuple[int, int] = (1, 4),
        input_channels: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.window_len = window_len
        self.embed_dim = embed_dim
        self.patch_size = tuple(patch_size)
        self.grid_channels = in_channels // self.patch_size[0]
        self.grid_steps = window_len // self.patch_size[1]
        self.num_tokens = self.grid_channels * self.grid_steps
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.input_proj = nn.Linear(input_channels, 1) if input_channels != 1 else None
        layers: list[nn.Module] = []
        head_schedule: list[int] = []
        for stage, depth in enumerate(depths):
            head_schedule.extend([int(num_heads[min(stage, len(num_heads) - 1)])] * int(depth))
        for heads in head_schedule:
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            layers.append(layer)
        self.blocks = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.bypass_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @classmethod
    def from_config(cls, cfg, *, input_channels: int = 1) -> "SwinEMGEncoder":
        return cls(
            in_channels=int(cfg.data.channels),
            window_len=int(round(cfg.data.sampling_rate * cfg.data.window_ms / 1000)),
            embed_dim=int(cfg.model.embed_dim),
            depths=list(cfg.model.depths),
            num_heads=list(cfg.model.num_heads),
            mlp_ratio=float(cfg.model.mlp_ratio),
            dropout=float(cfg.model.dropout),
            patch_size=tuple(cfg.model.patch_size),
            input_channels=input_channels,
        )

    def forward(self, x: torch.Tensor, *, return_tokens: bool = False, return_bypass: bool = False):
        if x.ndim == 3:
            x_grid = x.unsqueeze(1)
        elif x.ndim == 4:
            x_grid = x
        else:
            raise ValueError(f"Expected (B,C,W) or (B,F,C,W), got {tuple(x.shape)}")
        tokens = self.patch_embed(x_grid).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        raw_tokens = tokens
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        outputs = [pooled]
        if return_tokens:
            outputs.append(tokens)
        if return_bypass:
            bypass_seed = raw_tokens.mean(dim=1)
            outputs.append(self.bypass_proj(bypass_seed))
        return tuple(outputs) if len(outputs) > 1 else pooled

from __future__ import annotations

import torch
from torch import nn


class RawEMGBranch(nn.Module):
    def __init__(self, embed_dim: int, patch_size: tuple[int, int] = (1, 4), input_channels: int = 1):
        super().__init__()
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim == 3:
            raw_grid = raw.unsqueeze(1)
        elif raw.ndim == 4:
            raw_grid = raw
        else:
            raise ValueError(f"Expected raw EMG as (B,C,W) or (B,F,C,W), got {tuple(raw.shape)}")
        return self.patch_embed(raw_grid).flatten(2).transpose(1, 2)


class RawTokenFusion(nn.Module):
    def __init__(self, embed_dim: int, enabled: bool = True, patch_size: tuple[int, int] = (1, 4)):
        super().__init__()
        self.enabled = enabled
        self.raw_branch = RawEMGBranch(embed_dim, patch_size=patch_size)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        if not self.enabled or raw is None:
            return tokens
        raw_tokens = self.raw_branch(raw)
        if raw_tokens.shape[1] != tokens.shape[1]:
            raise ValueError(f"Raw bypass produced {raw_tokens.shape[1]} tokens, expected {tokens.shape[1]}")
        return self.norm(tokens + self.alpha * raw_tokens)


class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.3, bypass: bool = True, patch_size: tuple[int, int] = (1, 4)):
        super().__init__()
        self.fusion = RawTokenFusion(embed_dim, bypass, patch_size=patch_size)
        hidden = max(embed_dim // 2, 32)
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, *, tokens: torch.Tensor, pooled: torch.Tensor, bypass: torch.Tensor | None = None, raw: torch.Tensor | None = None) -> torch.Tensor:
        fused_tokens = self.fusion(tokens, raw)
        fused_pooled = fused_tokens.mean(dim=1) if self.fusion.enabled and raw is not None else pooled
        return self.net(fused_pooled)


class MambaHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        bypass: bool = True,
        bidirectional: bool = False,
        patch_size: tuple[int, int] = (1, 4),
        **mamba_kwargs,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError("head=mamba/bimamba requires official mamba-ssm on the server") from exc
        self.bidirectional = bidirectional
        self.fusion = RawTokenFusion(embed_dim, bypass, patch_size=patch_size)
        self.fwd = Mamba(d_model=embed_dim, **mamba_kwargs)
        self.rev = Mamba(d_model=embed_dim, **mamba_kwargs) if bidirectional else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, *, tokens: torch.Tensor, pooled: torch.Tensor, bypass: torch.Tensor | None = None, raw: torch.Tensor | None = None) -> torch.Tensor:
        fused_tokens = self.fusion(tokens, raw)
        fwd = self.fwd(fused_tokens).mean(dim=1)
        if self.rev is not None:
            rev = torch.flip(self.rev(torch.flip(fused_tokens, dims=[1])), dims=[1]).mean(dim=1)
            feat = torch.cat([fwd, rev], dim=-1)
        else:
            feat = fwd
        return self.fc(self.dropout(feat))


def build_head(name: str, embed_dim: int, num_classes: int, dropout: float = 0.3, bypass: bool = True, patch_size: tuple[int, int] = (1, 4), **kwargs) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPHead(embed_dim, num_classes, dropout=dropout, bypass=bypass, patch_size=patch_size)
    if name == "mamba":
        return MambaHead(embed_dim, num_classes, dropout=dropout, bypass=bypass, bidirectional=False, patch_size=patch_size, **kwargs)
    if name == "bimamba":
        return MambaHead(embed_dim, num_classes, dropout=dropout, bypass=bypass, bidirectional=True, patch_size=patch_size, **kwargs)
    raise ValueError(f"Unknown head: {name}")

from __future__ import annotations

import torch
from torch import nn


class ResidualBypass(nn.Module):
    def __init__(self, embed_dim: int, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, pooled: torch.Tensor, bypass: torch.Tensor | None) -> torch.Tensor:
        if not self.enabled or bypass is None:
            return self.norm(pooled)
        return self.norm(pooled + self.alpha * self.proj(bypass))


class MLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.3, bypass: bool = True):
        super().__init__()
        self.bypass = ResidualBypass(embed_dim, bypass)
        hidden = max(embed_dim // 2, 32)
        self.net = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, *, tokens: torch.Tensor, pooled: torch.Tensor, bypass: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(self.bypass(pooled, bypass))


class MambaHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.3, bypass: bool = True, bidirectional: bool = False, **mamba_kwargs):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            raise ImportError("head=mamba/bimamba requires official mamba-ssm on the server") from exc
        self.bidirectional = bidirectional
        self.bypass = ResidualBypass(embed_dim, bypass)
        self.fwd = Mamba(d_model=embed_dim, **mamba_kwargs)
        self.rev = Mamba(d_model=embed_dim, **mamba_kwargs) if bidirectional else None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, *, tokens: torch.Tensor, pooled: torch.Tensor, bypass: torch.Tensor | None = None) -> torch.Tensor:
        fwd = self.fwd(tokens).mean(dim=1)
        if self.rev is not None:
            rev = torch.flip(self.rev(torch.flip(tokens, dims=[1])), dims=[1]).mean(dim=1)
            feat = torch.cat([fwd, rev], dim=-1)
        else:
            feat = fwd
        if feat.size(-1) == pooled.size(-1):
            feat = self.bypass(feat, bypass)
        return self.fc(self.dropout(feat))


def build_head(name: str, embed_dim: int, num_classes: int, dropout: float = 0.3, bypass: bool = True, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPHead(embed_dim, num_classes, dropout=dropout, bypass=bypass)
    if name == "mamba":
        return MambaHead(embed_dim, num_classes, dropout=dropout, bypass=bypass, bidirectional=False, **kwargs)
    if name == "bimamba":
        return MambaHead(embed_dim, num_classes, dropout=dropout, bypass=bypass, bidirectional=True, **kwargs)
    raise ValueError(f"Unknown head: {name}")

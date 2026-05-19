from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tfmast.models.swin_emg import SwinEMGEncoder


@dataclass
class TFCOutput:
    losses: dict[str, torch.Tensor]


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0).float(), dim=-1)
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(2 * batch, dtype=torch.bool, device=z.device), -torch.finfo(sim.dtype).max)
    labels = torch.cat([torch.arange(batch, 2 * batch), torch.arange(0, batch)]).to(z.device)
    return F.cross_entropy(sim, labels)


def _time_augment(x: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(x) * 0.03
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.8, 1.2)
    shift = torch.randint(-2, 3, (x.size(0),), device=x.device)
    out = (x + noise) * scale
    return torch.stack([torch.roll(out[i], int(shift[i].item()), dims=-1) for i in range(x.size(0))], dim=0)


def _freq_features(x: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.rfft(x, dim=-1)
    amp = torch.log1p(spec.abs())
    phase = torch.angle(spec)
    feat = torch.stack([amp, phase], dim=1)
    return F.interpolate(feat, size=(x.size(1), x.size(2)), mode="bilinear", align_corners=False)


def _freq_augment(feat: torch.Tensor, alpha: float) -> torch.Tensor:
    out = feat.clone()
    bins = out.size(-1)
    if bins > 2:
        idx = torch.randint(1, bins, (out.size(0),), device=out.device)
        for b, i in enumerate(idx):
            out[b, 0, :, i] = out[b, 0, :, i] * alpha
    return out


class TFCModel(nn.Module):
    def __init__(self, time_encoder: SwinEMGEncoder, freq_encoder: SwinEMGEncoder, projector_dim: int, temperature: float, lam: float, margin: float, freq_alpha: float):
        super().__init__()
        self.time_encoder = time_encoder
        self.freq_encoder = freq_encoder
        dim = time_encoder.embed_dim
        self.time_projector = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, projector_dim))
        self.freq_projector = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, projector_dim))
        self.temperature = temperature
        self.lam = lam
        self.margin = margin
        self.freq_alpha = freq_alpha

    @classmethod
    def from_config(cls, cfg) -> "TFCModel":
        return cls(
            time_encoder=SwinEMGEncoder.from_config(cfg),
            freq_encoder=SwinEMGEncoder.from_config(cfg, input_channels=2),
            projector_dim=int(cfg.model.projector_dim),
            temperature=float(cfg.tfc.temperature),
            lam=float(cfg.tfc.lambda_contrastive),
            margin=float(cfg.tfc.margin),
            freq_alpha=float(cfg.tfc.freq_alpha),
        )

    def forward(self, x: torch.Tensor) -> TFCOutput:
        x_aug = _time_augment(x)
        f = _freq_features(x)
        f_aug = _freq_augment(f, self.freq_alpha)
        h_t = self.time_encoder(x)
        h_t_aug = self.time_encoder(x_aug)
        h_f = self.freq_encoder(f)
        h_f_aug = self.freq_encoder(f_aug)
        z_t = self.time_projector(h_t)
        z_t_aug = self.time_projector(h_t_aug)
        z_f = self.freq_projector(h_f)
        z_f_aug = self.freq_projector(h_f_aug)
        loss_time = _nt_xent(z_t, z_t_aug, self.temperature)
        loss_freq = _nt_xent(z_f, z_f_aug, self.temperature)
        d_pos = 1.0 - F.cosine_similarity(F.normalize(z_t, dim=-1), F.normalize(z_f, dim=-1), dim=-1)
        d_neg = 1.0 - F.cosine_similarity(F.normalize(z_t, dim=-1), F.normalize(z_f_aug, dim=-1), dim=-1)
        loss_consistency = F.relu(d_pos - d_neg + self.margin).mean()
        loss = self.lam * (loss_time + loss_freq) + (1.0 - self.lam) * loss_consistency
        sim = F.cosine_similarity(F.normalize(z_t, dim=-1), F.normalize(z_f, dim=-1), dim=-1).mean()
        return TFCOutput(losses={
            "loss": loss,
            "loss_time": loss_time,
            "loss_freq": loss_freq,
            "loss_consistency": loss_consistency,
            "embedding_similarity": sim,
        })

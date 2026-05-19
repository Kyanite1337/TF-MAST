from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from tfmast.models.swin_emg import SwinEMGEncoder
from tfmast.training.masks import MaskBank, expand_patch_mask


@dataclass
class MAEOutput:
    loss: torch.Tensor
    reconstruction: torch.Tensor
    mask: torch.Tensor


class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder: SwinEMGEncoder, mask_bank: MaskBank, decoder_depth: int = 2):
        super().__init__()
        self.encoder = encoder
        self.mask_bank = mask_bank
        self.patch_size = encoder.patch_size
        patch_values = self.patch_size[0] * self.patch_size[1]
        self.decoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=encoder.embed_dim,
                nhead=4 if encoder.embed_dim % 4 == 0 else 1,
                dim_feedforward=encoder.embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(encoder.embed_dim)
        self.to_patch = nn.Linear(encoder.embed_dim, patch_values)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, encoder.num_tokens, encoder.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    @classmethod
    def from_config(cls, cfg) -> "MaskedAutoencoder":
        encoder = SwinEMGEncoder.from_config(cfg)
        mask_bank = MaskBank(
            mask_ratio=float(cfg.mae.mask_ratio),
            strategies=list(cfg.mae.mask_strategies),
            patch_size=tuple(cfg.model.patch_size),
            decoder_mask_ratio=float(cfg.mae.decoder_mask_ratio),
        )
        return cls(encoder, mask_bank, decoder_depth=int(cfg.model.decoder_depth))

    def forward(self, x: torch.Tensor) -> MAEOutput:
        masks = self.mask_bank(x)
        sample_mask = expand_patch_mask(masks.encoder_mask, x.size(-1), self.patch_size)
        x_masked = x.masked_fill(sample_mask, 0.0)
        _, tokens, _ = self.encoder(x_masked, return_tokens=True, return_bypass=True)
        dec_mask = masks.decoder_mask.flatten(1)
        tokens = torch.where(dec_mask.unsqueeze(-1), self.mask_token.expand_as(tokens), tokens)
        decoded = tokens + self.decoder_pos_embed[:, : tokens.size(1), :]
        for block in self.decoder:
            decoded = block(decoded)
        decoded = self.decoder_norm(decoded)
        patch_values = self.to_patch(decoded)
        recon = self._unpatchify(patch_values, x.shape)
        loss_mask = expand_patch_mask(masks.decoder_mask, x.size(-1), self.patch_size)
        per_sample = F.mse_loss(recon, x, reduction="none")
        loss = per_sample[loss_mask].mean() if loss_mask.any() else per_sample.mean()
        return MAEOutput(loss=loss, reconstruction=recon, mask=loss_mask)

    def _unpatchify(self, patch_values: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        batch, channels, width = x_shape
        steps = width // self.patch_size[1]
        patches = patch_values.view(batch, channels, steps, self.patch_size[1])
        return patches.reshape(batch, channels, steps * self.patch_size[1])

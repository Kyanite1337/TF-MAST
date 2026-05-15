"""Stage 1 MAE Decoder: reverses Encoder's CNN stem via transposed convolutions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class Decoder(nn.Module):
    """Decodes latent vector back to original signal shape.

    Encoder path: 16→128 (s=1) → 256 (s=2) → 256 (s=2)  → latent dim 256
    Decoder path: 256 → 256×10 → 256 (s=2↑) → 128 (s=2↑) → 16
    """

    def __init__(self, latent_dim: int = 256, output_channels: int = 16, output_len: int = 40):
        super().__init__()
        self.output_len = output_len
        # CNN stem produces length // 4 = 10 for input length 40
        self.encoded_len = output_len // 4
        self.expand_dim = latent_dim

        self.fc = nn.Linear(latent_dim, latent_dim * self.encoded_len)

        self.deconv1 = nn.ConvTranspose1d(latent_dim, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.deconv2 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.proj = nn.Conv1d(128, output_channels, kernel_size=1)

    def forward(self, z):
        # z: (B, D)
        B = z.size(0)
        x = self.fc(z)  # (B, D * L_enc)
        x = x.view(B, self.expand_dim, self.encoded_len)  # (B, D, L_enc)
        x = F.gelu(self.bn1(self.deconv1(x)))  # (B, 256, L_enc*2)
        x = F.gelu(self.bn2(self.deconv2(x)))  # (B, 128, W)  — should match original length
        x = self.proj(x)  # (B, C, W)
        return x


def random_block_mask(x: torch.Tensor, mask_ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply random block masking to time dimension.

    Args:
        x:          (B, C, W)
        mask_ratio: fraction of time steps to mask

    Returns:
        x_masked: (B, C, W) with masked positions replaced by 0
        mask:     (B, 1, W) boolean, True = masked
    """
    B, C, W = x.shape
    device = x.device
    mask = torch.zeros(B, 1, W, device=device, dtype=torch.bool)

    n_mask = int(W * mask_ratio)
    for b in range(B):
        positions = list(range(W))
        masked_count = 0
        while masked_count < n_mask:
            blk_size = torch.randint(2, min(9, W + 1), (1,)).item()
            blk_size = min(blk_size, n_mask - masked_count)
            if len(positions) == 0:
                break
            start_idx = torch.randint(0, len(positions), (1,)).item()
            start = positions[start_idx]
            for i in range(blk_size):
                pos = start + i
                if pos < W and pos in positions:
                    mask[b, 0, pos] = True
                    positions.remove(pos)
                    masked_count += 1
                if masked_count >= n_mask:
                    break

    x_masked = x.clone()
    x_masked = x_masked * (~mask.expand(-1, C, -1))  # set masked to 0
    return x_masked, mask


def mae_loss(x: torch.Tensor, x_recon: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss only on masked positions."""
    loss = F.mse_loss(x_recon, x, reduction="none")
    # mask: (B, 1, W) → (B, C, W)
    mask_expanded = mask.expand(-1, x.size(1), -1)
    loss = loss[mask_expanded]
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=x.device)

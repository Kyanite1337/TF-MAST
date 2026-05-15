"""Stage 1: Masked Autoencoder pre-training.

Trains encoder to reconstruct randomly masked sEMG windows.
Saves encoder.pt for Stage 2.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import config as cfg
from models.encoder import Encoder
from models.decoder import Decoder, random_block_mask, mae_loss

try:
    import wandb
except ImportError:
    wandb = None


def build_mae():
    encoder = Encoder(
        in_channels=cfg.N_CHANNELS,
        conv_channels=cfg.ENC_CONV_CHANNELS,
        dim=cfg.ENC_DIM,
        num_layers=cfg.ENC_TRANSFORMER_LAYERS,
        heads=cfg.ENC_TRANSFORMER_HEADS,
        ffn_dim=cfg.ENC_TRANSFORMER_FFN,
        dropout=cfg.ENC_DROPOUT,
    )
    decoder = Decoder(
        latent_dim=cfg.ENC_DIM,
        output_channels=cfg.N_CHANNELS,
        output_len=cfg.WINDOW_LEN,
    )
    return encoder, decoder


def train_stage1(loader, encoder=None, decoder=None, epochs=None, device=None, save_dir=None,
                 wandb_run=None):
    """Train MAE. Returns (encoder, decoder)."""
    if device is None:
        device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    if epochs is None:
        epochs = cfg.MAE_EPOCHS
    if save_dir is None:
        save_dir = cfg.CHECKPOINT_DIR / "stage1_mae"
    save_dir.mkdir(parents=True, exist_ok=True)

    if encoder is None:
        encoder, decoder = build_mae()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.MAE_LR,
        weight_decay=cfg.MAE_WEIGHT_DECAY,
    )

    total_iters = len(loader) * epochs
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.MAE_WARMUP_EPOCHS * len(loader))
    cosine = CosineAnnealingLR(optimizer, T_max=total_iters - cfg.MAE_WARMUP_EPOCHS * len(loader))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.MAE_WARMUP_EPOCHS * len(loader)])

    print(f"[Stage 1 MAE] Training {epochs} epochs on {device}")
    encoder.train()
    decoder.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for x in loader:
            x = x.to(device)
            x_masked, mask = random_block_mask(x, cfg.MAE_MASK_RATIO)

            z = encoder(x_masked)
            x_recon = decoder(z)

            loss = mae_loss(x, x_recon, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        lr = scheduler.get_last_lr()[0]

        if wandb_run is not None and wandb is not None:
            wandb_run.log({
                "stage1/mae_loss": avg_loss,
                "stage1/lr": lr,
                "epoch": epoch + 1,
            })

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | MAE Loss: {avg_loss:.6f} | LR: {lr:.2e}")

    torch.save(encoder.state_dict(), str(save_dir / "encoder.pt"))
    torch.save(decoder.state_dict(), str(save_dir / "decoder.pt"))
    print(f"[Stage 1 MAE] Saved to {save_dir}")
    return encoder, decoder


if __name__ == "__main__":
    from preprocess import load_and_preprocess
    from dataset import get_loaders

    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    mae_loader, _, _, _ = get_loaders(X_train, y_train, X_test, y_test)
    train_stage1(mae_loader)

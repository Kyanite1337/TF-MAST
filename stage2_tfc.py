"""Stage 2: Time-Frequency Consistency pre-training with a single shared encoder.

Both time and frequency branches generate time-domain signals:
- Time branch:   raw → time augment → encoder → z_t, z_t_aug
- Freq branch:   raw → FFT → perturb → IFFT → encoder → z_f, z_f_aug

Loss: α(L_T + L_F) + (1-α)L_C
  L_T, L_F: NT-Xent contrastive (each branch's original vs augmented)
  L_C:      triplet consistency (cross-branch alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import config as cfg
from models.encoder import Encoder

try:
    import wandb
except ImportError:
    wandb = None


# ═══════════════════════════════════════════
# Augmentations
# ═══════════════════════════════════════════

def time_augment(x: torch.Tensor) -> torch.Tensor:
    B, C, W = x.shape
    device = x.device
    noise = torch.randn_like(x) * 0.05
    x = x + noise
    scale = torch.empty(B, 1, 1, device=device).uniform_(0.8, 1.2)
    x = x * scale
    shift = torch.randint(-5, 6, (B,), device=device)
    x_out = torch.zeros_like(x)
    for b in range(B):
        s = shift[b].item()
        if s > 0:
            x_out[b, :, s:] = x[b, :, :W - s]
        elif s < 0:
            x_out[b, :, :W + s] = x[b, :, -s:]
        else:
            x_out[b] = x[b]
    return x_out


def freq_augment(x: torch.Tensor) -> torch.Tensor:
    B, C, W = x.shape
    xf = torch.fft.rfft(x, dim=-1)
    n_bins = xf.shape[-1]
    for b in range(B):
        n_perturb = torch.randint(1, 3, (1,)).item()
        for _ in range(n_perturb):
            idx = torch.randint(1, n_bins, (1,)).item()
            xf[b, :, idx] = 0.0
    return torch.fft.irfft(xf, n=W, dim=-1)


# ═══════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=-1)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels], dim=0)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    return F.cross_entropy(sim, labels)


def triplet_consistency(z_t: torch.Tensor, z_f: torch.Tensor, z_f_aug: torch.Tensor,
                        margin: float = 1.0) -> torch.Tensor:
    z_t_n = F.normalize(z_t, dim=-1)
    z_f_n = F.normalize(z_f, dim=-1)
    z_f_aug_n = F.normalize(z_f_aug, dim=-1)
    d_pos = 1.0 - F.cosine_similarity(z_t_n, z_f_n, dim=-1)
    d_neg = 1.0 - F.cosine_similarity(z_t_n, z_f_aug_n, dim=-1)
    return F.relu(d_pos - d_neg + margin).mean()


# ═══════════════════════════════════════════
# Stage 2 training
# ═══════════════════════════════════════════

def build_tfc_encoder(pretrained_path: str = None) -> Encoder:
    encoder = Encoder(
        in_channels=cfg.N_CHANNELS,
        conv_channels=cfg.ENC_CONV_CHANNELS,
        dim=cfg.ENC_DIM,
        num_layers=cfg.ENC_TRANSFORMER_LAYERS,
        heads=cfg.ENC_TRANSFORMER_HEADS,
        ffn_dim=cfg.ENC_TRANSFORMER_FFN,
        dropout=cfg.ENC_DROPOUT,
    )
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu")
        encoder.load_state_dict(state)
        print(f"[Stage 2 TFC] Loaded encoder from {pretrained_path}")
    return encoder


def train_stage2(loader, encoder=None, epochs=None, device=None, save_dir=None,
                 wandb_run=None):
    if device is None:
        device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    if epochs is None:
        epochs = cfg.TFC_EPOCHS
    if save_dir is None:
        save_dir = cfg.CHECKPOINT_DIR / "stage2_tfc"
    save_dir.mkdir(parents=True, exist_ok=True)

    if encoder is None:
        encoder = build_tfc_encoder()
    encoder = encoder.to(device)
    encoder.train()

    optimizer = AdamW(encoder.parameters(), lr=cfg.TFC_LR, weight_decay=cfg.TFC_WEIGHT_DECAY)

    total_iters = len(loader) * epochs
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.TFC_WARMUP_EPOCHS * len(loader))
    cosine = CosineAnnealingLR(optimizer, T_max=total_iters - cfg.TFC_WARMUP_EPOCHS * len(loader))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.TFC_WARMUP_EPOCHS * len(loader)])

    alpha = cfg.TFC_ALPHA
    tau = cfg.TFC_TEMPERATURE
    margin = cfg.TFC_MARGIN

    print(f"[Stage 2 TFC] Training {epochs} epochs on {device}")

    for epoch in range(epochs):
        total_loss = total_lt = total_lf = total_lc = 0.0

        for x, _ in loader:
            x = x.to(device)

            # Time branch
            z_t = encoder(x)
            z_t_aug = encoder(time_augment(x))

            # Frequency branch
            x_f = freq_augment(x)
            x_f_aug = freq_augment(x)
            z_f = encoder(x_f)
            z_f_aug = encoder(x_f_aug)

            # Losses
            lt = nt_xent(z_t, z_t_aug, tau)
            lf = nt_xent(z_f, z_f_aug, tau)
            lc = triplet_consistency(z_t, z_f, z_f_aug, margin)

            loss = alpha * (lt + lf) + (1.0 - alpha) * lc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_lt += lt.item()
            total_lf += lf.item()
            total_lc += lc.item()

        n = len(loader)
        avg_loss = total_loss / n
        avg_lt = total_lt / n
        avg_lf = total_lf / n
        avg_lc = total_lc / n
        lr = scheduler.get_last_lr()[0]

        if wandb_run is not None and wandb is not None:
            wandb_run.log({
                "stage2/total_loss": avg_loss,
                "stage2/L_T": avg_lt,
                "stage2/L_F": avg_lf,
                "stage2/L_C": avg_lc,
                "stage2/lr": lr,
                "epoch": epoch + 1,
            })

        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} "
                  f"(L_T: {avg_lt:.4f}, L_F: {avg_lf:.4f}, L_C: {avg_lc:.4f}) "
                  f"LR: {lr:.2e}")

    torch.save(encoder.state_dict(), str(save_dir / "encoder.pt"))
    print(f"[Stage 2 TFC] Saved to {save_dir}")
    return encoder


if __name__ == "__main__":
    from preprocess import load_and_preprocess
    from dataset import get_loaders

    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    _, tfc_loader, _, _ = get_loaders(X_train, y_train, X_test, y_test)
    enc = build_tfc_encoder(pretrained_path=str(cfg.CHECKPOINT_DIR / "stage1_mae" / "encoder.pt"))
    train_stage2(tfc_loader, encoder=enc)

"""Stage 3: Supervised fine-tuning on labeled sEMG gesture data."""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import config as cfg
from models.encoder import Encoder
from models.classifier import Classifier

try:
    import wandb
except ImportError:
    wandb = None


def build_model(encoder_path: str = None):
    encoder = Encoder(
        in_channels=cfg.N_CHANNELS,
        conv_channels=cfg.ENC_CONV_CHANNELS,
        dim=cfg.ENC_DIM,
        num_layers=cfg.ENC_TRANSFORMER_LAYERS,
        heads=cfg.ENC_TRANSFORMER_HEADS,
        ffn_dim=cfg.ENC_TRANSFORMER_FFN,
        dropout=cfg.ENC_DROPOUT,
    )
    if encoder_path is not None:
        state = torch.load(encoder_path, map_location="cpu")
        encoder.load_state_dict(state)
        print(f"[Stage 3] Loaded encoder from {encoder_path}")

    classifier = Classifier(in_dim=cfg.ENC_DIM, num_classes=cfg.N_CLASSES, dropout=cfg.FT_DROPOUT)
    return encoder, classifier


def train_stage3(train_loader, test_loader, encoder_path=None, epochs=None, device=None,
                 save_dir=None, wandb_run=None):
    if device is None:
        device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    if epochs is None:
        epochs = cfg.FT_EPOCHS
    if save_dir is None:
        save_dir = cfg.CHECKPOINT_DIR / "stage3_finetune"
    save_dir.mkdir(parents=True, exist_ok=True)

    encoder, classifier = build_model(encoder_path)
    encoder = encoder.to(device)
    classifier = classifier.to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=cfg.FT_LR,
        weight_decay=cfg.FT_WEIGHT_DECAY,
    )

    total_iters = len(train_loader) * epochs
    warmup = LinearLR(optimizer, start_factor=0.1,
                      total_iters=cfg.FT_WARMUP_EPOCHS * len(train_loader))
    cosine = CosineAnnealingLR(optimizer,
                               T_max=total_iters - cfg.FT_WARMUP_EPOCHS * len(train_loader))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[cfg.FT_WARMUP_EPOCHS * len(train_loader)])

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.FT_LABEL_SMOOTHING)

    print(f"[Stage 3 Fine-tune] Training {epochs} epochs on {device}")
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        encoder.train()
        classifier.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            z = encoder(x)
            logits = classifier(z)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        lr = scheduler.get_last_lr()[0]

        # Evaluate every epoch, log to wandb
        acc, f1 = evaluate(encoder, classifier, test_loader, device)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(encoder.state_dict(), str(save_dir / "encoder_best.pt"))
            torch.save(classifier.state_dict(), str(save_dir / "classifier_best.pt"))

        if wandb_run is not None and wandb is not None:
            wandb_run.log({
                "stage3/train_loss": avg_loss,
                "stage3/test_acc": acc,
                "stage3/test_f1": f1,
                "stage3/lr": lr,
                "stage3/best_acc": best_acc,
                "epoch": epoch + 1,
            })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs} | Loss: {avg_loss:.4f} "
                  f"| Test Acc: {acc:.4f} | Test F1: {f1:.4f} | LR: {lr:.2e}")

    # Load best and final eval
    encoder.load_state_dict(torch.load(str(save_dir / "encoder_best.pt"), map_location=device))
    classifier.load_state_dict(torch.load(str(save_dir / "classifier_best.pt"), map_location=device))
    final_acc, final_f1, cm = evaluate(encoder, classifier, test_loader, device, verbose=True)

    # Log final confusion matrix to wandb
    if wandb_run is not None and wandb is not None:
        wandb_run.log({
            "stage3/final_acc": final_acc,
            "stage3/final_f1": final_f1,
            "stage3/best_epoch": best_epoch,
            "stage3/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=cm.astype(int).tolist() if hasattr(cm, 'tolist') else None,
                preds=None,
                class_names=[str(i) for i in range(cfg.N_CLASSES)],
            ),
        })
        # Log raw confusion matrix as a table
        cm_table = wandb.Table(
            columns=["class"] + [str(i) for i in range(cfg.N_CLASSES)],
            data=[[str(i)] + [int(v) for v in row] for i, row in enumerate(cm)]
        )
        wandb_run.log({"stage3/confusion_matrix_table": cm_table})

    print(f"\n[Stage 3] Best Epoch: {best_epoch} | Test Acc: {final_acc:.4f} | Test F1: {final_f1:.4f}")

    return encoder, classifier, {"accuracy": final_acc, "f1": final_f1, "best_epoch": best_epoch}


@torch.no_grad()
def evaluate(encoder, classifier, loader, device, verbose=False):
    encoder.eval()
    classifier.eval()

    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = encoder(x)
        logits = classifier(z)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    if verbose:
        print(f"  Confusion Matrix ({len(cm)}x{len(cm)}):")
        print(f"  {cm}")

    encoder.train()
    classifier.train()
    return acc, f1, cm


if __name__ == "__main__":
    from preprocess import load_and_preprocess
    from dataset import get_loaders

    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    _, _, train_loader, test_loader = get_loaders(X_train, y_train, X_test, y_test)
    train_stage3(train_loader, test_loader,
                 encoder_path=str(cfg.CHECKPOINT_DIR / "stage2_tfc" / "encoder.pt"))

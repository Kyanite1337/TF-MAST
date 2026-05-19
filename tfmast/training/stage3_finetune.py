from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW

from tfmast.models.heads import build_head
from tfmast.models.swin_emg import SwinEMGEncoder
from tfmast.training.common import TrainResult, append_metrics, gpu_memory_mb, make_run_dir, resolve_device, save_checkpoint, set_seed
from tfmast.utils.wandb import WandbLogger


def _load_encoder(encoder: SwinEMGEncoder, path) -> None:
    if path is None:
        return
    state = torch.load(path, map_location="cpu")
    encoder_state = state.get("encoder", state.get("model", state))
    encoder.load_state_dict(encoder_state, strict=False)


@torch.no_grad()
def evaluate(encoder, head, loader, device):
    encoder.eval()
    head.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        pooled, tokens, bypass = encoder(x, return_tokens=True, return_bypass=True)
        logits = head(tokens=tokens, pooled=pooled, bypass=bypass)
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(y.cpu())
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }


def train_finetune(cfg, train_loader, test_loader, *, init_encoder=None, head_name: str | None = None, run_name: str | None = None) -> TrainResult:
    set_seed(int(cfg.train.seed))
    device = resolve_device(cfg)
    run_dir = make_run_dir(cfg, "finetune", run_name)
    logger = WandbLogger(cfg, run_name=run_name or "finetune", stage="finetune")
    encoder = SwinEMGEncoder.from_config(cfg)
    _load_encoder(encoder, init_encoder)
    head_name = head_name or cfg.head.name
    head = build_head(
        head_name,
        embed_dim=int(cfg.model.embed_dim),
        num_classes=53 if cfg.data.class_mode == "53_with_rest" else 52,
        dropout=float(cfg.head.dropout),
        bypass=bool(cfg.head.bypass),
        d_state=int(cfg.head.mamba_d_state),
        d_conv=int(cfg.head.mamba_d_conv),
        expand=int(cfg.head.mamba_expand),
    )
    encoder, head = encoder.to(device), head.to(device)
    opt = AdamW(list(encoder.parameters()) + list(head.parameters()), lr=float(cfg.train.finetune.lr), weight_decay=float(cfg.train.finetune.weight_decay))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.train.finetune.label_smoothing))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    best = -1.0
    best_metrics = {}
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    max_batches = cfg.train.max_batches
    log_every = max(0, int(getattr(cfg.train, "log_every_steps", 0) or 0))
    for epoch in range(1, int(cfg.train.finetune.epochs) + 1):
        start = time.time()
        encoder.train()
        head.train()
        total = 0.0
        steps = 0
        accum = max(1, int(cfg.train.grad_accum_steps))
        opt.zero_grad(set_to_none=True)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=bool(cfg.train.amp and device.type == "cuda")):
                pooled, tokens, bypass = encoder(x, return_tokens=True, return_bypass=True)
                logits = head(tokens=tokens, pooled=pooled, bypass=bypass)
                loss = criterion(logits, y)
                scaled_loss = loss / accum
            scaler.scale(scaled_loss).backward()
            if (steps + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu())
            steps += 1
            if log_every and (steps == 1 or steps % log_every == 0):
                print(f"[FT] epoch {epoch:03d} step {steps}/{len(train_loader)} loss={float(loss.detach().cpu()):.6f}", flush=True)
            if max_batches and steps >= int(max_batches):
                break
        if steps % accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        eval_metrics = evaluate(encoder, head, test_loader, device)
        train_loss = total / max(steps, 1)
        metrics = {"epoch": epoch, "train_loss": train_loss, "finetune/train_loss": train_loss, "finetune/accuracy": eval_metrics["accuracy"], "finetune/macro_f1": eval_metrics["macro_f1"], "lr": opt.param_groups[0]["lr"], "epoch_time": time.time() - start, "gpu_memory_mb": gpu_memory_mb()}
        print(
            f"[FT] epoch {epoch:03d} loss={train_loss:.6f} "
            f"acc={eval_metrics['accuracy']:.4f} macro_f1={eval_metrics['macro_f1']:.4f} "
            f"time={metrics['epoch_time']:.1f}s",
            flush=True,
        )
        append_metrics(run_dir, metrics | {"confusion_matrix": eval_metrics["confusion_matrix"]})
        logger.log(metrics, step=epoch)
        payload = {"encoder": encoder.state_dict(), "head": head.state_dict(), "metrics": metrics, "confusion_matrix": eval_metrics["confusion_matrix"]}
        save_checkpoint(last_path, **payload)
        if eval_metrics["macro_f1"] > best:
            best = eval_metrics["macro_f1"]
            best_metrics = {"accuracy": eval_metrics["accuracy"], "macro_f1": eval_metrics["macro_f1"]}
            save_checkpoint(best_path, **payload)
    logger.finish()
    return TrainResult(run_dir=run_dir, best_checkpoint=best_path, last_checkpoint=last_path, best_metrics=best_metrics)

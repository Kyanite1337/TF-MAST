from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim import AdamW

from tfmast.models.mae import MaskedAutoencoder
from tfmast.training.common import EarlyStopper, TrainResult, append_metrics, gpu_memory_mb, make_run_dir, resolve_device, save_checkpoint, set_seed
from tfmast.utils.wandb import WandbLogger


@torch.no_grad()
def _evaluate_mae(model: MaskedAutoencoder, loader, device: torch.device, max_batches=None) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for batch in loader:
        out = model(batch.to(device))
        total += float(out.loss.detach().cpu())
        steps += 1
        if max_batches and steps >= int(max_batches):
            break
    return total / max(steps, 1)


def train_mae(cfg, loader, val_loader=None, *, run_name: str | None = None) -> TrainResult:
    set_seed(int(cfg.train.seed))
    device = resolve_device(cfg)
    run_dir = make_run_dir(cfg, "mae", run_name)
    logger = WandbLogger(cfg, run_name=run_name or "mae", stage="mae")
    model = MaskedAutoencoder.from_config(cfg).to(device)
    opt = AdamW(model.parameters(), lr=float(cfg.train.mae.lr), weight_decay=float(cfg.train.mae.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    stopper = EarlyStopper(mode="min", patience=int(cfg.train.mae.early_stop_patience), min_delta=float(cfg.train.mae.early_stop_min_delta))
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    max_batches = cfg.train.max_batches
    log_every = max(0, int(getattr(cfg.train, "log_every_steps", 0) or 0))
    for epoch in range(1, int(cfg.train.mae.epochs) + 1):
        start = time.time()
        model.train()
        total = 0.0
        steps = 0
        accum = max(1, int(cfg.train.grad_accum_steps))
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            x = batch.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=bool(cfg.train.amp and device.type == "cuda")):
                out = model(x)
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if (steps + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += float(out.loss.detach().cpu())
            steps += 1
            if log_every and (steps == 1 or steps % log_every == 0):
                print(f"[MAE] epoch {epoch:03d} step {steps}/{len(loader)} loss={float(out.loss.detach().cpu()):.6f}", flush=True)
            if max_batches and steps >= int(max_batches):
                break
        if steps % accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        train_loss = total / max(steps, 1)
        val_loss = _evaluate_mae(model, val_loader or loader, device, max_batches=max_batches)
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "mae/reconstruction_loss": train_loss, "mae/val_loss": val_loss, "lr": opt.param_groups[0]["lr"], "epoch_time": time.time() - start, "gpu_memory_mb": gpu_memory_mb()}
        print(f"MAE Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {metrics['epoch_time']:.1f}s", flush=True)
        append_metrics(run_dir, metrics)
        logger.log(metrics, step=epoch)
        payload = {"model": model.state_dict(), "encoder": model.encoder.state_dict(), "metrics": metrics}
        save_checkpoint(last_path, **payload)
        decision = stopper.update(val_loss)
        if decision.improved:
            save_checkpoint(best_path, **payload)
            print(f"  └── Saved best MAE (loss: {val_loss:.4f})", flush=True)
        if decision.should_stop:
            print("  └── MAE Early stopping triggered.", flush=True)
            break
    logger.finish()
    return TrainResult(run_dir=run_dir, best_checkpoint=best_path, last_checkpoint=last_path, best_metrics={"loss": stopper.best})

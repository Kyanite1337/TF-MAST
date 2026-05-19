from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim import AdamW

from tfmast.models.mae import MaskedAutoencoder
from tfmast.training.common import TrainResult, append_metrics, gpu_memory_mb, make_run_dir, resolve_device, save_checkpoint, set_seed
from tfmast.utils.wandb import WandbLogger


def train_mae(cfg, loader, *, run_name: str | None = None) -> TrainResult:
    set_seed(int(cfg.train.seed))
    device = resolve_device(cfg)
    run_dir = make_run_dir(cfg, "mae", run_name)
    logger = WandbLogger(cfg, run_name=run_name or "mae", stage="mae")
    model = MaskedAutoencoder.from_config(cfg).to(device)
    opt = AdamW(model.parameters(), lr=float(cfg.train.mae.lr), weight_decay=float(cfg.train.mae.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    best = float("inf")
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
        loss = total / max(steps, 1)
        metrics = {"epoch": epoch, "train_loss": loss, "mae/reconstruction_loss": loss, "lr": opt.param_groups[0]["lr"], "epoch_time": time.time() - start, "gpu_memory_mb": gpu_memory_mb()}
        print(f"[MAE] epoch {epoch:03d} loss={loss:.6f} lr={metrics['lr']:.3e} time={metrics['epoch_time']:.1f}s", flush=True)
        append_metrics(run_dir, metrics)
        logger.log(metrics, step=epoch)
        payload = {"model": model.state_dict(), "encoder": model.encoder.state_dict(), "metrics": metrics}
        save_checkpoint(last_path, **payload)
        if loss < best:
            best = loss
            save_checkpoint(best_path, **payload)
    logger.finish()
    return TrainResult(run_dir=run_dir, best_checkpoint=best_path, last_checkpoint=last_path, best_metrics={"loss": best})

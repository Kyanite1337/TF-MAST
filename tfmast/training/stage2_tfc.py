from __future__ import annotations

import time

import torch
from torch.optim import AdamW

from tfmast.models.tfc import TFCModel
from tfmast.training.common import EarlyStopper, TrainResult, append_metrics, gpu_memory_mb, make_run_dir, resolve_device, save_checkpoint, set_seed
from tfmast.utils.wandb import WandbLogger


def _load_tfc_init(model: TFCModel, path) -> None:
    if path is None:
        return
    state = torch.load(path, map_location="cpu")
    if "model" in state:
        current = model.state_dict()
        compatible = {k: v for k, v in state["model"].items() if k in current and current[k].shape == v.shape}
        skipped = sorted(k for k, v in state["model"].items() if k in current and current[k].shape != v.shape)
        model.load_state_dict(compatible, strict=False)
        if skipped:
            print(f"[init] skipped shape-mismatched TFC weights: {', '.join(skipped)}", flush=True)
        return
    encoder_state = state.get("encoder", state)
    model.time_encoder.load_state_dict(encoder_state, strict=False)


@torch.no_grad()
def _evaluate_tfc(model: TFCModel, loader, device: torch.device, max_batches=None) -> float:
    model.eval()
    total = 0.0
    steps = 0
    for batch in loader:
        out = model(batch.to(device))
        total += float(out.losses["loss"].detach().cpu())
        steps += 1
        if max_batches and steps >= int(max_batches):
            break
    return total / max(steps, 1)


def train_tfc(cfg, loader, val_loader=None, *, init_encoder=None, run_name: str | None = None) -> TrainResult:
    set_seed(int(cfg.train.seed))
    device = resolve_device(cfg)
    run_dir = make_run_dir(cfg, "tfc", run_name)
    logger = WandbLogger(cfg, run_name=run_name or "tfc", stage="tfc")
    model = TFCModel.from_config(cfg)
    _load_tfc_init(model, init_encoder)
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=float(cfg.train.tfc.lr), weight_decay=float(cfg.train.tfc.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    stopper = EarlyStopper(mode="min", patience=int(cfg.train.tfc.early_stop_patience), min_delta=float(cfg.train.tfc.early_stop_min_delta))
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    max_batches = cfg.train.max_batches
    log_every = max(0, int(getattr(cfg.train, "log_every_steps", 0) or 0))
    for epoch in range(1, int(cfg.train.tfc.epochs) + 1):
        start = time.time()
        total = {"loss": 0.0, "loss_time": 0.0, "loss_freq": 0.0, "loss_consistency": 0.0, "embedding_similarity": 0.0}
        steps = 0
        model.train()
        accum = max(1, int(cfg.train.grad_accum_steps))
        opt.zero_grad(set_to_none=True)
        for batch in loader:
            x = batch.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=bool(cfg.train.amp and device.type == "cuda")):
                out = model(x)
                loss = out.losses["loss"] / accum
            scaler.scale(loss).backward()
            if (steps + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            for key in total:
                total[key] += float(out.losses[key].detach().cpu())
            steps += 1
            if log_every and (steps == 1 or steps % log_every == 0):
                print(
                    f"[TFC] epoch {epoch:03d} step {steps}/{len(loader)} "
                    f"loss={float(out.losses['loss'].detach().cpu()):.6f}",
                    flush=True,
                )
            if max_batches and steps >= int(max_batches):
                break
        if steps % accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        avg = {key: value / max(steps, 1) for key, value in total.items()}
        val_loss = _evaluate_tfc(model, val_loader or loader, device, max_batches=max_batches)
        metrics = {"epoch": epoch, "train_loss": avg["loss"], "val_loss": val_loss, "tfc/total_loss": avg["loss"], "tfc/val_loss": val_loss, "tfc/L_time": avg["loss_time"], "tfc/L_freq": avg["loss_freq"], "tfc/L_consistency": avg["loss_consistency"], "tfc/embedding_similarity": avg["embedding_similarity"], "lr": opt.param_groups[0]["lr"], "epoch_time": time.time() - start, "gpu_memory_mb": gpu_memory_mb()}
        print(f"TFC Epoch {epoch:3d} | Train: {avg['loss']:.4f} | Val: {val_loss:.4f} | Time: {metrics['epoch_time']:.1f}s", flush=True)
        append_metrics(run_dir, metrics)
        logger.log(metrics, step=epoch)
        payload = {"model": model.state_dict(), "encoder": model.time_encoder.state_dict(), "metrics": metrics}
        save_checkpoint(last_path, **payload)
        decision = stopper.update(val_loss)
        if decision.improved:
            save_checkpoint(best_path, **payload)
            print(f"  └── Saved best TFC (loss: {val_loss:.4f})", flush=True)
        if decision.should_stop:
            print("  └── TFC Early stopping triggered.", flush=True)
            break
    logger.finish()
    return TrainResult(run_dir=run_dir, best_checkpoint=best_path, last_checkpoint=last_path, best_metrics={"loss": stopper.best})

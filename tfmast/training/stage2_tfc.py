from __future__ import annotations

import time

import torch
from torch.optim import AdamW

from tfmast.models.tfc import TFCModel
from tfmast.training.common import TrainResult, append_metrics, gpu_memory_mb, make_run_dir, resolve_device, save_checkpoint, set_seed
from tfmast.utils.wandb import WandbLogger


def _load_time_encoder(model: TFCModel, path) -> None:
    if path is None:
        return
    state = torch.load(path, map_location="cpu")
    encoder_state = state.get("encoder", state.get("model", state))
    model.time_encoder.load_state_dict(encoder_state, strict=False)


def train_tfc(cfg, loader, *, init_encoder=None, run_name: str | None = None) -> TrainResult:
    set_seed(int(cfg.train.seed))
    device = resolve_device(cfg)
    run_dir = make_run_dir(cfg, "tfc", run_name)
    logger = WandbLogger(cfg, run_name=run_name or "tfc", stage="tfc")
    model = TFCModel.from_config(cfg)
    _load_time_encoder(model, init_encoder)
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=float(cfg.train.tfc.lr), weight_decay=float(cfg.train.tfc.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.train.amp and device.type == "cuda"))
    best = float("inf")
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    max_batches = cfg.train.max_batches
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
            if max_batches and steps >= int(max_batches):
                break
        if steps % accum != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        avg = {key: value / max(steps, 1) for key, value in total.items()}
        metrics = {"epoch": epoch, "train_loss": avg["loss"], "tfc/total_loss": avg["loss"], "tfc/L_time": avg["loss_time"], "tfc/L_freq": avg["loss_freq"], "tfc/L_consistency": avg["loss_consistency"], "tfc/embedding_similarity": avg["embedding_similarity"], "lr": opt.param_groups[0]["lr"], "epoch_time": time.time() - start, "gpu_memory_mb": gpu_memory_mb()}
        print(f"[TFC] epoch {epoch:03d} loss={avg['loss']:.6f} Lt={avg['loss_time']:.4f} Lf={avg['loss_freq']:.4f} Lc={avg['loss_consistency']:.4f}")
        append_metrics(run_dir, metrics)
        logger.log(metrics, step=epoch)
        payload = {"model": model.state_dict(), "encoder": model.time_encoder.state_dict(), "metrics": metrics}
        save_checkpoint(last_path, **payload)
        if avg["loss"] < best:
            best = avg["loss"]
            save_checkpoint(best_path, **payload)
    logger.finish()
    return TrainResult(run_dir=run_dir, best_checkpoint=best_path, last_checkpoint=last_path, best_metrics={"loss": best})

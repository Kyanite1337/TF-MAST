"""Run full 3-stage pre-training pipeline + ablation experiments on NinaPro DB5.

Usage:
    python run_all.py                        # full pipeline (MAE → TFC → Fine-tune)
    python run_all.py --ablation             # all 4 ablation variants
    python run_all.py --stage 1              # run only Stage 1
    python run_all.py --stage 2              # run only Stage 2 (needs stage1 encoder)
    python run_all.py --stage 3              # run only Stage 3
    python run_all.py --stage 2 --from-scratch   # TFC from random init
    python run_all.py --stage 3 --from-scratch   # Fine-tune from random init
    python run_all.py --stage 3 --encoder stage1_mae
    python run_all.py --wandb-offline           # use offline wandb mode
    python run_all.py --no-wandb                # disable wandb entirely
"""

import argparse
import random
import numpy as np
import torch

import config as cfg
from preprocess import load_and_preprocess
from dataset import get_loaders

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_wandb(run_name: str, group: str = None, tags: list = None,
                wandb_mode: str = None):
    """Initialize a wandb run with hyperparameter config."""
    if wandb is None:
        return None
    if wandb_mode is None:
        wandb_mode = cfg.WANDB_MODE
    if wandb_mode == "disabled":
        return None

    run = wandb.init(
        project=cfg.WANDB_PROJECT,
        entity=cfg.WANDB_ENTITY,
        name=run_name,
        group=group,
        tags=tags,
        mode=wandb_mode,
        config={
            # Dataset
            "n_classes": cfg.N_CLASSES,
            "n_channels": cfg.N_CHANNELS,
            "window_ms": cfg.WINDOW_MS,
            "stride_ms": cfg.STRIDE_MS,
            "window_len": cfg.WINDOW_LEN,
            # Encoder
            "enc_dim": cfg.ENC_DIM,
            "enc_transformer_layers": cfg.ENC_TRANSFORMER_LAYERS,
            "enc_transformer_heads": cfg.ENC_TRANSFORMER_HEADS,
            "enc_ffn_dim": cfg.ENC_TRANSFORMER_FFN,
            "enc_dropout": cfg.ENC_DROPOUT,
            # Stage 1
            "mae_mask_ratio": cfg.MAE_MASK_RATIO,
            "mae_epochs": cfg.MAE_EPOCHS,
            "mae_batch_size": cfg.MAE_BATCH_SIZE,
            "mae_lr": cfg.MAE_LR,
            "mae_weight_decay": cfg.MAE_WEIGHT_DECAY,
            # Stage 2
            "tfc_epochs": cfg.TFC_EPOCHS,
            "tfc_batch_size": cfg.TFC_BATCH_SIZE,
            "tfc_lr": cfg.TFC_LR,
            "tfc_alpha": cfg.TFC_ALPHA,
            "tfc_temperature": cfg.TFC_TEMPERATURE,
            "tfc_margin": cfg.TFC_MARGIN,
            # Stage 3
            "ft_epochs": cfg.FT_EPOCHS,
            "ft_batch_size": cfg.FT_BATCH_SIZE,
            "ft_lr": cfg.FT_LR,
            "ft_label_smoothing": cfg.FT_LABEL_SMOOTHING,
            # General
            "seed": cfg.SEED,
        },
    )
    return run


def _finish_wandb(run):
    if run is not None and wandb is not None:
        run.finish()


def run_full_pipeline(wandb_mode=None):
    """Run MAE → TFC → Fine-tune sequentially."""
    run = _init_wandb("full-pipeline", group="pipeline", tags=["full", "mae+tfc+ft"],
                      wandb_mode=wandb_mode)

    print("=" * 60)
    print("FULL PIPELINE: Stage 1 (MAE) → Stage 2 (TFC) → Stage 3 (Fine-tune)")
    print("=" * 60)

    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    mae_loader, tfc_loader, train_loader, test_loader = get_loaders(
        X_train, y_train, X_test, y_test)

    # Stage 1: MAE
    print("\n" + "=" * 40)
    print("STAGE 1: MAE Pretraining")
    print("=" * 40)
    from stage1_mae import train_stage1
    encoder, decoder = train_stage1(mae_loader, wandb_run=run)
    stage1_encoder_path = str(cfg.CHECKPOINT_DIR / "stage1_mae" / "encoder.pt")

    # Stage 2: TFC
    print("\n" + "=" * 40)
    print("STAGE 2: TFC Pretraining")
    print("=" * 40)
    from stage2_tfc import build_tfc_encoder, train_stage2
    encoder = build_tfc_encoder(pretrained_path=stage1_encoder_path)
    encoder = train_stage2(tfc_loader, encoder=encoder, wandb_run=run)
    stage2_encoder_path = str(cfg.CHECKPOINT_DIR / "stage2_tfc" / "encoder.pt")

    # Stage 3: Fine-tune
    print("\n" + "=" * 40)
    print("STAGE 3: Fine-tuning")
    print("=" * 40)
    from stage3_finetune import train_stage3
    _, _, metrics = train_stage3(train_loader, test_loader,
                                 encoder_path=stage2_encoder_path, wandb_run=run)

    if run is not None:
        run.summary["final_acc"] = metrics["accuracy"]
        run.summary["final_f1"] = metrics["f1"]

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    print("=" * 60)

    _finish_wandb(run)
    return metrics


def run_ablation(wandb_mode=None):
    """Run all 4 ablation experiments with separate wandb runs."""
    print("=" * 60)
    print("ABLATION STUDY: comparing all 4 training configurations")
    print("=" * 60)

    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    mae_loader, tfc_loader, train_loader, test_loader = get_loaders(
        X_train, y_train, X_test, y_test)

    results = {}

    # ── M0: Random init → Fine-tune ──
    print("\n" + "-" * 40)
    print("M0: Random Init → Fine-tune (baseline)")
    print("-" * 40)
    run0 = _init_wandb("M0_random_FT", group="ablation", tags=["M0", "random", "no-pretrain"],
                       wandb_mode=wandb_mode)
    from stage3_finetune import train_stage3
    _, _, m0 = train_stage3(train_loader, test_loader, encoder_path=None,
                            save_dir=cfg.CHECKPOINT_DIR / "ablation_m0", wandb_run=run0)
    if run0 is not None:
        run0.summary["final_acc"] = m0["accuracy"]
        run0.summary["final_f1"] = m0["f1"]
        run0.finish()
    results["M0 (random→FT)"] = m0

    # ── M1: MAE → Fine-tune ──
    print("\n" + "-" * 40)
    print("M1: MAE → Fine-tune")
    print("-" * 40)
    run1 = _init_wandb("M1_MAE_FT", group="ablation", tags=["M1", "mae", "no-tfc"],
                       wandb_mode=wandb_mode)
    from stage1_mae import train_stage1
    train_stage1(mae_loader, wandb_run=run1)
    _, _, m1 = train_stage3(train_loader, test_loader,
                            encoder_path=str(cfg.CHECKPOINT_DIR / "stage1_mae" / "encoder.pt"),
                            save_dir=cfg.CHECKPOINT_DIR / "ablation_m1", wandb_run=run1)
    if run1 is not None:
        run1.summary["final_acc"] = m1["accuracy"]
        run1.summary["final_f1"] = m1["f1"]
        run1.finish()
    results["M1 (MAE→FT)"] = m1

    # ── M2: TFC (random init) → Fine-tune ──
    print("\n" + "-" * 40)
    print("M2: TFC (from scratch) → Fine-tune")
    print("-" * 40)
    run2 = _init_wandb("M2_TFC_FT", group="ablation", tags=["M2", "tfc", "no-mae"],
                       wandb_mode=wandb_mode)
    from stage2_tfc import build_tfc_encoder, train_stage2
    encoder = build_tfc_encoder(pretrained_path=None)
    train_stage2(tfc_loader, encoder=encoder, wandb_run=run2)
    _, _, m2 = train_stage3(train_loader, test_loader,
                            encoder_path=str(cfg.CHECKPOINT_DIR / "stage2_tfc" / "encoder.pt"),
                            save_dir=cfg.CHECKPOINT_DIR / "ablation_m2", wandb_run=run2)
    if run2 is not None:
        run2.summary["final_acc"] = m2["accuracy"]
        run2.summary["final_f1"] = m2["f1"]
        run2.finish()
    results["M2 (TFC→FT)"] = m2

    # ── M3: Full pipeline (MAE → TFC → Fine-tune) ──
    print("\n" + "-" * 40)
    print("M3: MAE → TFC → Fine-tune (full)")
    print("-" * 40)
    run3 = _init_wandb("M3_MAE_TFC_FT", group="ablation", tags=["M3", "full", "mae+tfc+ft"],
                       wandb_mode=wandb_mode)
    from stage2_tfc import build_tfc_encoder as build_tfc
    encoder = build_tfc(pretrained_path=str(cfg.CHECKPOINT_DIR / "stage1_mae" / "encoder.pt"))
    train_stage2(tfc_loader, encoder=encoder, wandb_run=run3)
    _, _, m3 = train_stage3(train_loader, test_loader,
                            encoder_path=str(cfg.CHECKPOINT_DIR / "stage2_tfc" / "encoder.pt"),
                            save_dir=cfg.CHECKPOINT_DIR / "ablation_m3", wandb_run=run3)
    if run3 is not None:
        run3.summary["final_acc"] = m3["accuracy"]
        run3.summary["final_f1"] = m3["f1"]
        run3.finish()
    results["M3 (MAE→TFC→FT)"] = m3

    # ── Log summary comparison table ──
    summary_run = _init_wandb("ablation-summary", group="ablation",
                              tags=["summary"], wandb_mode=wandb_mode)
    if summary_run is not None:
        rows = [[name, m["accuracy"], m["f1"]] for name, m in results.items()]
        tbl = wandb.Table(columns=["Experiment", "Accuracy", "F1 Score"], data=rows)
        summary_run.log({"ablation_summary": tbl})
        for name, m in results.items():
            summary_run.log({f"summary/{name}_acc": m["accuracy"],
                            f"summary/{name}_f1": m["f1"]})
        summary_run.finish()

    # Console summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:25s} | Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3])
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.no_wandb:
        wandb_mode = "disabled"
    elif args.wandb_offline:
        wandb_mode = "offline"
    else:
        wandb_mode = cfg.WANDB_MODE

    print(f"[INFO] Device: {cfg.DEVICE}, Seed: {args.seed}, Wandb: {wandb_mode}")

    if args.ablation:
        run_ablation(wandb_mode=wandb_mode)
        return

    if args.stage is None:
        run_full_pipeline(wandb_mode=wandb_mode)
        return

    # Single stage
    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    mae_loader, tfc_loader, train_loader, test_loader = get_loaders(
        X_train, y_train, X_test, y_test)

    if args.stage == 1:
        run = _init_wandb("stage1-MAE", group="single-stage", tags=["stage1", "mae"],
                          wandb_mode=wandb_mode)
        from stage1_mae import train_stage1
        train_stage1(mae_loader, wandb_run=run)
        _finish_wandb(run)

    elif args.stage == 2:
        tag = "stage2-from-scratch" if args.from_scratch else "stage2-from-MAE"
        run = _init_wandb(f"stage2-TFC_{tag}", group="single-stage",
                          tags=["stage2", "tfc"], wandb_mode=wandb_mode)
        from stage2_tfc import build_tfc_encoder, train_stage2
        path = None if args.from_scratch else str(cfg.CHECKPOINT_DIR / "stage1_mae" / "encoder.pt")
        enc = build_tfc_encoder(pretrained_path=path)
        train_stage2(tfc_loader, encoder=enc, wandb_run=run)
        _finish_wandb(run)

    elif args.stage == 3:
        if args.encoder:
            path = str(cfg.CHECKPOINT_DIR / args.encoder / "encoder.pt")
            tag = f"stage3-from-{args.encoder}"
        elif args.from_scratch:
            path = None
            tag = "stage3-from-scratch"
        else:
            path = str(cfg.CHECKPOINT_DIR / "stage2_tfc" / "encoder.pt")
            tag = "stage3-from-TFC"
        run = _init_wandb(f"stage3-FT_{tag}", group="single-stage",
                          tags=["stage3", "finetune"], wandb_mode=wandb_mode)
        from stage3_finetune import train_stage3
        train_stage3(train_loader, test_loader, encoder_path=path, wandb_run=run)
        _finish_wandb(run)


if __name__ == "__main__":
    main()

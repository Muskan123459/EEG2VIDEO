"""Training engine: dataloader construction, the training loop, and checkpoint I/O.

``train_group4_pipeline`` runs the full training schedule with checkpoint resume,
a cosine-annealing LR schedule with linear warmup, per-epoch CSV logging, and
periodic + best-validation checkpoints.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .config import CFG, DEVICE
from .utils import list_files
from .data import (
    VideoFeatureExtractor,
    paired_indices,
    split_pairs,
    EEGVideoPairDataset,
    collate_pairs,
)
from .pipeline import Group4Pipeline


def build_dataloaders(clip_extractor: "VideoFeatureExtractor") -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    eeg_files = list_files(CFG.eeg_root, ".npy")
    # Accept common video extensions and also accept directories with frames
    video_files = list_files(CFG.video_root, ".mp4") + list_files(CFG.video_root, ".avi") + list_files(CFG.video_root, ".mov")
    pairs = paired_indices(eeg_files, video_files)
    if len(pairs) == 0:
        return None, None, None

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, CFG.dataset_split_seed, CFG.val_ratio, CFG.test_ratio)
    # Share a single CLIP extractor across all splits to save ~1.8 GB GPU memory
    train_ds = EEGVideoPairDataset(train_pairs, clip_extractor)
    val_ds = EEGVideoPairDataset(val_pairs, clip_extractor) if val_pairs else None
    test_ds = EEGVideoPairDataset(test_pairs, clip_extractor) if test_pairs else None

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pairs)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pairs) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pairs) if test_ds else None
    return train_loader, val_loader, test_loader


def train_group4_pipeline():
    pipeline = Group4Pipeline(CFG)
    train_loader, val_loader, test_loader = build_dataloaders(pipeline.clip_extractor)
    if train_loader is None:
        print("Warning: No EEG/video pairs found. Skipping training.")
        return pipeline

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    ckpt_path = CFG.output_dir / "group4_best_checkpoint.pt"
    start_epoch = 0
    best_val = float("inf")

    if ckpt_path.exists():
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        pipeline.st2.load_state_dict(ckpt["st2"])
        pipeline.st3.load_state_dict(ckpt["st3"])
        pipeline.st4.load_state_dict(ckpt["st4"])
        pipeline.st5.load_state_dict(ckpt["st5"])
        pipeline.st6.load_state_dict(ckpt["st6"])
        pipeline.optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"  → Resuming from epoch {start_epoch}, best_val={best_val:.4f}")

    # ── Cosine annealing LR scheduler with linear warmup ──────────────────────
    # Warmup: linearly ramp from lr*0.1 → lr over warmup_epochs
    # Then:   cosine decay from lr → lr * lr_min_factor over remaining epochs
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < CFG.warmup_epochs:
            # linear warmup
            return 0.1 + 0.9 * (current_epoch / max(CFG.warmup_epochs, 1))
        # cosine decay
        progress = (current_epoch - CFG.warmup_epochs) / max(
            CFG.epochs - CFG.warmup_epochs, 1
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return CFG.lr_min_factor + (1.0 - CFG.lr_min_factor) * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        pipeline.optimizer, lr_lambda=lr_lambda, last_epoch=start_epoch - 1
    )

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_path = CFG.output_dir / "training_log.csv"
    write_header = not log_path.exists()
    log_file = open(log_path, "a")
    if write_header:
        log_file.write(
            "epoch,loss,l_mae,l_contrastive,l_clip,l_diffusion,l_temporal,"
            "l_recon,val_loss_proxy,val_clip_sim,lr_encoder,lr_unet\n"
        )

    print(f"=== Group 4 Training ===")
    print(f"Device      : {DEVICE}")
    print(f"Start epoch : {start_epoch + 1} / {CFG.epochs}")
    print(
        f"Train / Val / Test : "
        f"{len(train_loader.dataset)} / "
        f"{len(val_loader.dataset) if val_loader else 0} / "
        f"{len(test_loader.dataset) if test_loader else 0}"
    )

    # ── Main loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG.epochs):

        train_logs = pipeline.train_one_epoch(train_loader, epoch)

        # Current LRs (two param groups: encoder, unet)
        current_lrs = [pg["lr"] for pg in pipeline.optimizer.param_groups]
        lr_enc  = current_lrs[0] if len(current_lrs) > 0 else CFG.lr_encoder
        lr_unet = current_lrs[1] if len(current_lrs) > 1 else CFG.lr_unet

        print(
            f"Epoch {epoch+1:>4}/{CFG.epochs} | "
            f"Loss={train_logs['loss']:.4f} | "
            f"MAE={train_logs['l_mae']:.4f} | "
            f"Contrast={train_logs['l_contrastive']:.4f} | "
            f"CLIP={train_logs['l_clip']:.4f} | "
            f"Diff={train_logs['l_diffusion']:.4f} | "
            f"Temp={train_logs['l_temporal']:.4f} | "
            f"Recon={train_logs['l_recon']:.4f} | "
            f"LR_enc={lr_enc:.2e} | LR_unet={lr_unet:.2e}"
        )

        val_loss_proxy = float("inf")
        val_clip_sim   = 0.0

        if val_loader is not None:
            val_logs       = pipeline.validate(val_loader)
            val_loss_proxy = val_logs["val_loss_proxy"]
            val_clip_sim   = val_logs["val_clip_sim"]
            print(
                f"  → Validation | proxy={val_loss_proxy:.4f} | "
                f"clip_sim={val_clip_sim:.4f}"
            )

            if val_loss_proxy < best_val:
                best_val = val_loss_proxy
                torch.save(
                    {
                        "st2":       pipeline.st2.state_dict(),
                        "st3":       pipeline.st3.state_dict(),
                        "st4":       pipeline.st4.state_dict(),
                        "st5":       pipeline.st5.state_dict(),
                        "st6":       pipeline.st6.state_dict(),
                        "optimizer": pipeline.optimizer.state_dict(),
                        "epoch":     epoch + 1,        # ← saved so resume works
                        "best_val":  best_val,
                    },
                    ckpt_path,
                )
                print(f"  ✓ Best checkpoint saved (val={best_val:.4f})")

        # ── Periodic checkpoint every 50 epochs (safe fallback) ───────────────
        if (epoch + 1) % 50 == 0:
            periodic_path = CFG.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "st2":       pipeline.st2.state_dict(),
                    "st3":       pipeline.st3.state_dict(),
                    "st4":       pipeline.st4.state_dict(),
                    "st5":       pipeline.st5.state_dict(),
                    "st6":       pipeline.st6.state_dict(),
                    "optimizer": pipeline.optimizer.state_dict(),
                    "epoch":     epoch + 1,
                    "best_val":  best_val,
                },
                periodic_path,
            )
            print(f"  ✓ Periodic checkpoint saved → {periodic_path}")

        # ── CSV log ───────────────────────────────────────────────────────────
        log_file.write(
            f"{epoch+1},{train_logs['loss']:.6f},{train_logs['l_mae']:.6f},"
            f"{train_logs['l_contrastive']:.6f},{train_logs['l_clip']:.6f},"
            f"{train_logs['l_diffusion']:.6f},{train_logs['l_temporal']:.6f},"
            f"{train_logs['l_recon']:.6f},{val_loss_proxy:.6f},"
            f"{val_clip_sim:.6f},{lr_enc:.8f},{lr_unet:.8f}\n"
        )
        log_file.flush()

        # Step scheduler at end of each epoch
        scheduler.step()

    log_file.close()
    print(f"\nTraining complete. Log saved to: {log_path}")
    return pipeline


def load_checkpoint(pipeline: Group4Pipeline, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    pipeline.st2.load_state_dict(ckpt["st2"])
    pipeline.st3.load_state_dict(ckpt["st3"])
    pipeline.st4.load_state_dict(ckpt["st4"])
    pipeline.st5.load_state_dict(ckpt["st5"])
    pipeline.st6.load_state_dict(ckpt["st6"])
    if "optimizer" in ckpt:
        pipeline.optimizer.load_state_dict(ckpt["optimizer"])

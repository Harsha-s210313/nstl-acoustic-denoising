"""
detector_train.py
-----------------
Stage 1 training pipeline for SignalDetectorCNN.

This script is self-contained and runnable directly:
    python detector_train.py

NOTE: The user already has a working detector. This file is provided for
completeness and reproducibility.  If you want to retrain the detector,
populate:
    data/detector/signal/   (*.dat, label=1)
    data/detector/noise/    (*.dat, label=0)
and run this script.

Workflow
--------
1. Load Config (single source of truth).
2. Seed everything for reproducibility.
3. Build DetectorDataset (train + val splits).
4. Instantiate SignalDetectorCNN from Config.
5. Train with CrossEntropyLoss + Adam.
6. Validate each epoch; save best checkpoint.
7. Early stopping if validation loss does not improve.
"""

import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import DetectorDataset
from detector import SignalDetectorCNN
from utils import ensure_dir, set_seed

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
) -> tuple:
    """Run one epoch.  Returns (avg_loss, avg_accuracy)."""
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += _accuracy(logits, y)
            n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #

def train_detector(cfg: Config) -> None:
    """
    Full training pipeline for SignalDetectorCNN.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    """
    set_seed(cfg.RANDOM_SEED, cfg.DETERMINISTIC)
    ensure_dir(cfg.MODELS_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------ #
    # Datasets / DataLoaders
    # ------------------------------------------------------------------ #
    log.info("Loading detector datasets from: %s", cfg.DETECTOR_DATA_DIR)
    train_ds = DetectorDataset(cfg, split="train", augment=True)
    val_ds = DetectorDataset(cfg, split="val", augment=False)

    if len(train_ds) == 0:
        raise RuntimeError(
            "Detector training set is empty.  Populate "
            f"{cfg.DETECTOR_DATA_DIR}/signal and .../noise with .dat files."
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.DETECTOR_BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.DETECTOR_BATCH_SIZE, shuffle=False
    )

    # ------------------------------------------------------------------ #
    # Model, loss, optimizer
    # ------------------------------------------------------------------ #
    model = SignalDetectorCNN(cfg).to(device)
    log.info(
        "SignalDetectorCNN: features=%s  kernel=%d  dropout=%.2f  fc_hidden=%d",
        cfg.DETECTOR_FEATURES, cfg.DETECTOR_KERNEL_SIZE,
        cfg.DETECTOR_DROPOUT, cfg.DETECTOR_FC_HIDDEN,
    )

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable parameters: %d", n_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.DETECTOR_LR,
        weight_decay=cfg.DETECTOR_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6, verbose=False
    )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.DETECTOR_EPOCHS + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )

        if len(val_ds) > 0:
            val_loss, val_acc = _run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )
        else:
            val_loss, val_acc = float("nan"), float("nan")

        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d | lr=%.2e | "
            "train loss=%.4f acc=%.3f | val loss=%.4f acc=%.3f",
            epoch, cfg.DETECTOR_EPOCHS, current_lr,
            tr_loss, tr_acc, val_loss, val_acc,
        )

        if not isinstance(val_loss, float) or not (val_loss != val_loss):  # not NaN
            scheduler.step(val_loss)

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "cfg_features": cfg.DETECTOR_FEATURES,
                    "cfg_kernel": cfg.DETECTOR_KERNEL_SIZE,
                    "cfg_fc_hidden": cfg.DETECTOR_FC_HIDDEN,
                    "cfg_signal_length": cfg.SIGNAL_LENGTH,
                }
                torch.save(checkpoint, cfg.DETECTOR_CHECKPOINT)
                log.info("  → Saved best checkpoint (val_loss=%.4f)", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= cfg.DETECTOR_PATIENCE:
                    log.info(
                        "Early stopping: no improvement for %d epochs.",
                        cfg.DETECTOR_PATIENCE,
                    )
                    break

    log.info("Detector training complete.  Best val_loss=%.4f", best_val_loss)
    log.info("Checkpoint saved to: %s", cfg.DETECTOR_CHECKPOINT)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    cfg = Config()
    train_detector(cfg)

"""
Training script for Stage 1 CNN Pulse Detector.
Optimizes binary classification (pulse present vs ambient noise).
"""

import os
import argparse
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from .config import Config, default_config
    from .utils import set_seed, ensure_dir
    from .dataset import DetectorDataset
    from .detector import SignalDetectorCNN
except (ImportError, ValueError):
    from config import Config, default_config
    from utils import set_seed, ensure_dir
    from dataset import DetectorDataset
    from detector import SignalDetectorCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_detector(
    cfg: Optional[Config] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    patience: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train SignalDetectorCNN using supervised cross-entropy loss.
    Saves best model to cfg.DETECTOR_CHECKPOINT with metadata.
    """
    config = cfg or default_config
    config.validate()
    set_seed(config.RANDOM_SEED, config.DETERMINISTIC)

    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)
    logger.info("Detector training device: %s", dev)

    num_epochs = epochs if epochs is not None else config.DETECTOR_EPOCHS
    b_size = batch_size if batch_size is not None else config.DETECTOR_BATCH_SIZE
    learning_rate = lr if lr is not None else config.DETECTOR_LR
    early_patience = patience if patience is not None else config.DETECTOR_PATIENCE

    train_ds = DetectorDataset(config, split="train", augment=True)
    val_ds = DetectorDataset(config, split="val", augment=False)

    if len(train_ds) == 0:
        raise RuntimeError(f"Detector dataset is empty in {config.DETECTOR_DATA_DIR}. Generate data first.")

    train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, drop_last=False)
    has_val = len(val_ds) > 0
    val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=False) if has_val else None

    model = SignalDetectorCNN(cfg=config).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.DETECTOR_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    logger.info("Starting Detector training: %d epochs, batch_size=%d, lr=%.2e", num_epochs, b_size, learning_rate)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        # Validation
        if has_val and val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(dev), y.to(dev)
                    logits = model(x)
                    loss = criterion(logits, y)
                    val_loss += loss.item() * len(y)
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

            val_loss /= len(val_ds)
            scheduler.step(val_loss)

            preds_arr = np.array(all_preds)
            targets_arr = np.array(all_targets)
            acc = float(np.mean(preds_arr == targets_arr))
            tp = int(np.sum((preds_arr == 1) & (targets_arr == 1)))
            fp = int(np.sum((preds_arr == 1) & (targets_arr == 0)))
            fn = int(np.sum((preds_arr == 0) & (targets_arr == 1)))
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            logger.info(
                "Epoch %02d/%02d - Train Loss: %.4f, Val Loss: %.4f, Acc: %.4f, Prec: %.4f, Rec: %.4f",
                epoch, num_epochs, train_loss, val_loss, acc, precision, recall
            )

            monitored_loss = val_loss
        else:
            monitored_loss = train_loss
            acc, precision, recall = 1.0, 1.0, 1.0
            logger.info("Epoch %02d/%02d - Train Loss: %.4f (No Val)", epoch, num_epochs, train_loss)

        if monitored_loss < best_val_loss:
            best_val_loss = monitored_loss
            patience_counter = 0
            best_metrics = {"accuracy": acc, "precision": precision, "recall": recall}

            ensure_dir(config.DETECTOR_CHECKPOINT)
            checkpoint = {
                "tag": "detector",
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_metrics": best_metrics,
                "model_state_dict": model.state_dict(),
                "config": {
                    "features": config.DETECTOR_FEATURES,
                    "kernel_size": config.DETECTOR_KERNEL_SIZE,
                    "fc_hidden": config.DETECTOR_FC_HIDDEN,
                },
            }
            torch.save(checkpoint, config.DETECTOR_CHECKPOINT)
            logger.info("Saved best detector checkpoint to %s (Loss: %.4f)", config.DETECTOR_CHECKPOINT, best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= early_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    logger.info("Detector training complete. Best Val Loss: %.4f, Metrics: %s", best_val_loss, best_metrics)
    return {"val_loss": best_val_loss, "metrics": best_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 Acoustic Pulse Detector")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    train_detector(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )

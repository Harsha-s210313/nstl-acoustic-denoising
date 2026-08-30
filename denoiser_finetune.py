"""
denoiser_finetune.py
--------------------
Canonical Stage 2 training implementation.

Contains TWO classes:
  SyntheticDenoiserTrainer  — Stage 2A supervised pretraining on synthetic pairs
  TrialDataFineTuner        — Stage 2B unsupervised fine-tuning on real trial data

Entry-point (--stage argument required):
  python denoiser_finetune.py --stage pretrain     # runs Stage 2A
  python denoiser_finetune.py --stage finetune     # runs Stage 2B

If --stage is omitted the script defaults to the SAFER action (pretrain)
and prints a reminder that manual verification is expected before fine-tuning.

Workflow safety
---------------
The intended workflow is:
    Stage 2A pretrain  →  MANUAL VERIFICATION  →  Stage 2B finetune

This file enforces that:
  - Fine-tuning will not run without --stage finetune being explicitly given.
  - Fine-tuning requires the pretrained checkpoint to exist; it will refuse
    to start from scratch with an untrained model.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import Config
from dataset import UNetPretrainDataset, NoiseAdaptationUNetDataset
from unet import Configurable1DUNet
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
# Unsupervised noise-suppression loss
# --------------------------------------------------------------------------- #

class UnsupervisedNoiseSuppressionLoss(nn.Module):
    """
    Penalise the denoiser output for retaining noise energy.

    The loss has two terms:

    1. **Noise-energy term**: The output is correlated with the noise reference
       by comparing the mean squared amplitude of (output − noisy_input).
       We want the denoiser to move the signal *away* from the noisy input in
       the direction that reduces noise energy.

       loss_noise = mean((denoised - noisy)^2)
           — this penalises large deviations; on its own it would push the
             denoiser toward the identity.  Combined with the spectral term
             below it biases the solution toward smoother outputs.

    2. **Spectral smoothness term**: Penalise high-frequency energy in the
       output (a soft noise filter prior for underwater acoustic signals).

       loss_smooth = mean(diff(denoised)^2)

    Total loss = lambda * loss_noise + (1 - lambda) * loss_smooth

    where lambda = Config.FINETUNE_NOISE_LAMBDA.

    NOTE: This is an unsupervised objective.  It does NOT require clean
    ground-truth signals.  The noise_profile returned by
    NoiseAdaptationUNetDataset is passed in for future extensions (e.g.,
    noise-matched spectral subtraction); the current implementation uses
    the structural loss only.

    Parameters
    ----------
    lam : float
        Weight on the noise-energy term (0 ≤ lam ≤ 1).
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(
        self,
        denoised: torch.Tensor,
        noisy: torch.Tensor,
        noise_profile: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        denoised      : (B, 1, L)   — U-Net output
        noisy         : (B, 1, L)   — raw trial input
        noise_profile : (B, 1, W)   — noise reference segment (W ≤ L)
                        Not directly used in the loss computation here but
                        available for future spectral-subtraction extensions.

        Returns
        -------
        torch.Tensor : scalar loss
        """
        # Term 1: how much the output differs from the noisy input
        # (penalises the identity solution; combined with term 2 it shapes
        # the denoiser toward smoother, lower-noise outputs)
        diff = denoised - noisy
        loss_noise = torch.mean(diff ** 2)

        # Term 2: penalise sharp transitions in the output (smoothness prior)
        grad = denoised[:, :, 1:] - denoised[:, :, :-1]  # finite differences
        loss_smooth = torch.mean(grad ** 2)

        # Noise-profile-based regularisation: the denoised output should not
        # contain energy significantly larger than the noise floor.
        # Estimate noise power from the reference window.
        noise_power = torch.mean(noise_profile ** 2, dim=-1, keepdim=True)  # (B,1,1)
        signal_power = torch.mean(denoised ** 2, dim=-1, keepdim=True)      # (B,1,1)
        # Soft penalty: denoised power should exceed noise power (otherwise the
        # denoiser is just suppressing everything).  We penalise denoised power
        # that is LOWER than the noise floor (avoids over-suppression).
        over_suppression = torch.clamp(noise_power - signal_power, min=0.0)
        loss_over_suppress = torch.mean(over_suppression)

        total = self.lam * loss_noise + (1.0 - self.lam) * loss_smooth + 0.1 * loss_over_suppress
        return total


# --------------------------------------------------------------------------- #
# Checkpoint utilities
# --------------------------------------------------------------------------- #

def _save_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int,
    val_loss: float,
    cfg: Config,
    tag: str,
) -> None:
    """Save a model checkpoint with enough metadata to detect mismatches."""
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(
        {
            "tag": tag,
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "unet_config": cfg.UNET_CONFIG,
            "signal_length": cfg.SIGNAL_LENGTH,
        },
        path,
    )


def _load_checkpoint(
    path: str,
    model: nn.Module,
    cfg: Config,
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint, validate configuration against the current Config,
    and populate the model's state_dict.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    RuntimeError
        If the stored UNET_CONFIG or signal_length mismatches the current Config.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path!r}. "
            "Run the pretraining stage first: python denoiser_finetune.py --stage pretrain"
        )

    ckpt = torch.load(path, map_location="cpu")

    # Validate configuration consistency
    stored_cfg = ckpt.get("unet_config")
    if stored_cfg is not None:
        if stored_cfg.get("features") != cfg.UNET_CONFIG.get("features"):
            raise RuntimeError(
                f"Checkpoint feature widths {stored_cfg.get('features')} do not "
                f"match current UNET_CONFIG features {cfg.UNET_CONFIG.get('features')}. "
                "Update Config.UNET_CONFIG to match the checkpoint or retrain."
            )
    stored_len = ckpt.get("signal_length")
    if stored_len is not None and stored_len != cfg.SIGNAL_LENGTH:
        raise RuntimeError(
            f"Checkpoint SIGNAL_LENGTH={stored_len} does not match "
            f"Config.SIGNAL_LENGTH={cfg.SIGNAL_LENGTH}."
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    log.info(
        "Loaded checkpoint %r (tag=%r, epoch=%d, val_loss=%.6f)",
        path,
        ckpt.get("tag", "unknown"),
        ckpt.get("epoch", -1),
        ckpt.get("val_loss", float("nan")),
    )
    return ckpt


# --------------------------------------------------------------------------- #
# Stage 2A: Synthetic pretraining
# --------------------------------------------------------------------------- #

class SyntheticDenoiserTrainer:
    """
    Stage 2A: Supervised U-Net pretraining on synthetic clean/noisy pairs.

    Uses MSE loss.  Saves a checkpoint to Config.UNET_PRETRAINED_CHECKPOINT.

    Parameters
    ----------
    cfg : Config
        Configuration object (single source of truth).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.RANDOM_SEED, cfg.DETERMINISTIC)
        ensure_dir(cfg.MODELS_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("[SyntheticDenoiserTrainer] Device: %s", self.device)

    def run(self) -> None:
        """Execute Stage 2A pretraining."""
        cfg = self.cfg

        # ------------------------------------------------------------------ #
        # Datasets
        # ------------------------------------------------------------------ #
        log.info(
            "Loading synthetic dataset from: %s (SYNTHETIC_IS_COMPLEX=%s)",
            cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_IS_COMPLEX,
        )
        train_ds = UNetPretrainDataset(cfg, split="train", augment=True)
        val_ds = UNetPretrainDataset(cfg, split="val", augment=False)

        if len(train_ds) == 0:
            raise RuntimeError(
                "Synthetic training set is empty.  Populate "
                f"{cfg.SYNTHETIC_DATA_DIR}/clean and .../noisy with matched .dat files."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.PRETRAIN_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.PRETRAIN_BATCH_SIZE,
            shuffle=False,
        ) if len(val_ds) > 0 else None

        log.info(
            "Train batches: %d | Val samples: %d",
            len(train_loader), len(val_ds),
        )

        # ------------------------------------------------------------------ #
        # Model
        # ------------------------------------------------------------------ #
        model = Configurable1DUNet(cfg).to(self.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(
            "Configurable1DUNet: features=%s  kernel=%d  dropout=%.2f  params=%d",
            cfg.UNET_CONFIG["features"],
            cfg.UNET_CONFIG["kernel_size"],
            cfg.UNET_CONFIG["dropout"],
            n_params,
        )

        # ------------------------------------------------------------------ #
        # Optimizer, scheduler, loss
        # ------------------------------------------------------------------ #
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.PRETRAIN_LR,
            weight_decay=cfg.PRETRAIN_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.PRETRAIN_LR_FACTOR,
            patience=cfg.PRETRAIN_LR_PATIENCE,
            min_lr=cfg.PRETRAIN_LR_MIN,
            verbose=False,
        )
        criterion = nn.MSELoss()

        # ------------------------------------------------------------------ #
        # Training loop
        # ------------------------------------------------------------------ #
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.PRETRAIN_EPOCHS + 1):
            # --- Train ---
            model.train()
            tr_loss = 0.0
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                pred = model(noisy)
                loss = criterion(pred, clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= max(len(train_loader), 1)

            # --- Validate ---
            val_loss: Optional[float] = None
            if val_loader is not None and len(val_ds) > 0:
                model.eval()
                running_val = 0.0
                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy, clean = noisy.to(self.device), clean.to(self.device)
                        pred = model(noisy)
                        running_val += criterion(pred, clean).item()
                val_loss = running_val / max(len(val_loader), 1)
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            if val_loss is not None:
                log.info(
                    "Epoch %3d/%d | lr=%.2e | train MSE=%.6f | val MSE=%.6f",
                    epoch, cfg.PRETRAIN_EPOCHS, current_lr, tr_loss, val_loss,
                )
            else:
                log.info(
                    "Epoch %3d/%d | lr=%.2e | train MSE=%.6f | (no val split)",
                    epoch, cfg.PRETRAIN_EPOCHS, current_lr, tr_loss,
                )

            # --- Checkpoint & early stopping ---
            monitor = val_loss if val_loss is not None else tr_loss
            if monitor < best_val_loss:
                best_val_loss = monitor
                patience_counter = 0
                if cfg.PRETRAIN_SAVE_BEST:
                    _save_checkpoint(
                        cfg.UNET_PRETRAINED_CHECKPOINT,
                        model, epoch, monitor, cfg,
                        tag="pretrain",
                    )
                    log.info(
                        "  → Saved pretrained checkpoint (val_loss=%.6f)", monitor
                    )
            else:
                patience_counter += 1
                if patience_counter >= cfg.PRETRAIN_PATIENCE:
                    log.info(
                        "Early stopping after %d epochs without improvement.",
                        cfg.PRETRAIN_PATIENCE,
                    )
                    break

        # Always save final model (may not be best)
        final_path = cfg.UNET_PRETRAINED_CHECKPOINT.replace(".pth", "_final.pth")
        _save_checkpoint(
            final_path, model, cfg.PRETRAIN_EPOCHS,
            best_val_loss, cfg, tag="pretrain_final",
        )
        log.info(
            "[SyntheticDenoiserTrainer] Done. Best monitor loss=%.6f", best_val_loss
        )
        log.info(
            "Pretrained checkpoint: %s", cfg.UNET_PRETRAINED_CHECKPOINT
        )
        log.info(
            "\n⚠  MANUAL VERIFICATION REQUIRED before running fine-tuning.\n"
            "   Inspect the pretrained model on synthetic validation signals.\n"
            "   When satisfied, run:\n"
            "     python denoiser_finetune.py --stage finetune\n"
        )


# --------------------------------------------------------------------------- #
# Stage 2B: Trial fine-tuning
# --------------------------------------------------------------------------- #

class TrialDataFineTuner:
    """
    Stage 2B: Unsupervised U-Net fine-tuning on real trial data.

    Requires the pretrained checkpoint (from Stage 2A) to exist.
    Will refuse to run without it.

    Uses UnsupervisedNoiseSuppressionLoss.
    Saves a SEPARATE fine-tuned checkpoint (Config.UNET_FINETUNED_CHECKPOINT).
    The pretrained checkpoint is NEVER overwritten.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.RANDOM_SEED, cfg.DETERMINISTIC)
        ensure_dir(cfg.MODELS_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("[TrialDataFineTuner] Device: %s", self.device)

    def run(self) -> None:
        """Execute Stage 2B unsupervised fine-tuning."""
        cfg = self.cfg

        # ------------------------------------------------------------------ #
        # Load pretrained model (mandatory)
        # ------------------------------------------------------------------ #
        model = Configurable1DUNet(cfg).to(self.device)
        _load_checkpoint(cfg.UNET_PRETRAINED_CHECKPOINT, model, cfg)

        # ------------------------------------------------------------------ #
        # Trial dataset
        # ------------------------------------------------------------------ #
        log.info(
            "Loading trial dataset from: %s (TRIAL_IS_COMPLEX=%s)",
            cfg.TRIAL_DATA_DIR, cfg.TRIAL_IS_COMPLEX,
        )
        trial_ds = NoiseAdaptationUNetDataset(cfg, subset="all")
        trial_loader = DataLoader(
            trial_ds,
            batch_size=cfg.FINETUNE_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )
        log.info(
            "Trial dataset: %d files | %d batches/epoch",
            len(trial_ds), len(trial_loader),
        )

        # ------------------------------------------------------------------ #
        # Optimizer, scheduler, loss
        # ------------------------------------------------------------------ #
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.FINETUNE_LR,
            weight_decay=cfg.FINETUNE_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.FINETUNE_LR_FACTOR,
            patience=cfg.FINETUNE_LR_PATIENCE,
            min_lr=cfg.FINETUNE_LR_MIN,
            verbose=False,
        )
        criterion = UnsupervisedNoiseSuppressionLoss(lam=cfg.FINETUNE_NOISE_LAMBDA)

        # ------------------------------------------------------------------ #
        # Fine-tuning loop
        # ------------------------------------------------------------------ #
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, cfg.FINETUNE_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for noisy, noise_prof in trial_loader:
                noisy = noisy.to(self.device)
                noise_prof = noise_prof.to(self.device)

                denoised = model(noisy)
                loss = criterion(denoised, noisy, noise_prof)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)
            scheduler.step(epoch_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            log.info(
                "Epoch %3d/%d | lr=%.2e | finetune loss=%.6f",
                epoch, cfg.FINETUNE_EPOCHS, current_lr, epoch_loss,
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                _save_checkpoint(
                    cfg.UNET_FINETUNED_CHECKPOINT,
                    model, epoch, epoch_loss, cfg,
                    tag="finetune",
                )
                log.info(
                    "  → Saved fine-tuned checkpoint (loss=%.6f)", epoch_loss
                )
            else:
                patience_counter += 1
                if patience_counter >= cfg.FINETUNE_PATIENCE:
                    log.info(
                        "Early stopping after %d epochs without improvement.",
                        cfg.FINETUNE_PATIENCE,
                    )
                    break

        log.info(
            "[TrialDataFineTuner] Done. Best fine-tune loss=%.6f", best_loss
        )
        log.info("Fine-tuned checkpoint: %s", cfg.UNET_FINETUNED_CHECKPOINT)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "NSTL Acoustic Denoiser — Stage 2 training.\n\n"
            "Stages:\n"
            "  pretrain  Stage 2A: supervised pretraining on synthetic pairs\n"
            "  finetune  Stage 2B: unsupervised fine-tuning on real trial data\n\n"
            "IMPORTANT: Always manually verify the pretrained model before fine-tuning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["pretrain", "finetune"],
        default=None,
        help=(
            "Which training stage to run.  "
            "Defaults to 'pretrain' (safer default)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()

    if args.stage is None:
        log.warning(
            "--stage was not specified.  Defaulting to 'pretrain' (the safer choice).\n"
            "To run trial fine-tuning, explicitly pass --stage finetune AFTER\n"
            "manually verifying the pretrained model.\n"
        )
        args.stage = "pretrain"

    if args.stage == "pretrain":
        trainer = SyntheticDenoiserTrainer(cfg)
        trainer.run()

    elif args.stage == "finetune":
        log.info(
            "=== Stage 2B: Trial Fine-Tuning ===\n"
            "This step assumes you have MANUALLY VERIFIED the pretrained model.\n"
            "Starting fine-tuning in 3 seconds...\n"
        )
        import time
        time.sleep(3)
        finetuner = TrialDataFineTuner(cfg)
        finetuner.run()

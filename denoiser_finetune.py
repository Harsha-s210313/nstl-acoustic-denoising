"""
Canonical implementation of Stage 2 Denoising:
- UnsupervisedNoiseSuppressionLoss: Physically grounded energy matching loss
- SyntheticDenoiserTrainer: Stage 2A Supervised MSE pretraining on synthetic clean/noisy pairs
- TrialDataFineTuner: Stage 2B Unsupervised noise-reference fine-tuning on real trial data
"""

import os
import sys
import time
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
    from .dataset import UNetPretrainDataset, NoiseAdaptationUNetDataset
    from .unet import Configurable1DUNet
except (ImportError, ValueError):
    from config import Config, default_config
    from utils import set_seed, ensure_dir
    from dataset import UNetPretrainDataset, NoiseAdaptationUNetDataset
    from unet import Configurable1DUNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class UnsupervisedNoiseSuppressionLoss(nn.Module):
    """
    Unsupervised loss driven by acoustic energy balance:
      noisy_power = mean(noisy^2)
      noise_power = mean(noise_ref^2)
      signal_power = mean(denoised^2)
      target_power = clamp(noisy_power - noise_power, min=1e-8)
      loss_energy = mean((signal_power - target_power)^2)
      loss_smooth = mean((denoised[t+1] - denoised[t])^2) * smooth_scale
      loss_phase  = 1.0 - cosine_similarity(denoised, noisy)
      loss_over   = mean(clamp(noise_power - signal_power, min=0))
      total = lam * loss_energy + (1 - lam) * loss_smooth + phase_weight * loss_phase + 0.1 * loss_over

    Guaranteed properties:
    - NOT minimized by identity (denoised == noisy produces non-zero loss when noise_ref is present).
    - NOT minimized by all-zeros silence (signal_power == 0 incurs penalty from both target_power and loss_over).
    - Preserves high-frequency chirp carrier phase and waveform coherence, preventing correlation peak attenuation.
    """

    def __init__(
        self,
        lam: float = 0.7,
        smooth_scale: float = 0.05,
        phase_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.smooth_scale = float(smooth_scale)
        self.phase_weight = float(phase_weight)

    def forward(
        self,
        denoised: torch.Tensor,
        noisy: torch.Tensor,
        noise_ref: torch.Tensor,
    ) -> torch.Tensor:
        # Per-sample power estimates averaged over the time dimension (dim=-1)
        noisy_power = torch.mean(noisy ** 2, dim=-1, keepdim=True)
        noise_power = torch.mean(noise_ref ** 2, dim=-1, keepdim=True)
        signal_power = torch.mean(denoised ** 2, dim=-1, keepdim=True)

        target_power = torch.clamp(noisy_power - noise_power, min=1e-8)
        loss_energy = torch.mean((signal_power - target_power) ** 2)

        # Smoothness penalty scaled to prevent suppressing high-frequency acoustic chirps
        diff = denoised[..., 1:] - denoised[..., :-1]
        loss_smooth = torch.mean(diff ** 2) * self.smooth_scale

        # Carrier phase and waveform coherence with input signal:
        # Prevents phase drift and distortion of the LFM chirp trajectory
        dot = torch.sum(denoised * noisy, dim=-1, keepdim=True)
        norm_prod = torch.sqrt(torch.sum(denoised**2, dim=-1, keepdim=True) * torch.sum(noisy**2, dim=-1, keepdim=True) + 1e-8)
        loss_phase = torch.mean(1.0 - (dot / norm_prod))

        # Penalty against over-suppression below background noise floor
        loss_over = torch.mean(torch.clamp(noise_power - signal_power, min=0.0))

        total_loss = (
            self.lam * loss_energy
            + (1.0 - self.lam) * loss_smooth
            + self.phase_weight * loss_phase
            + 0.1 * loss_over
        )
        return total_loss


class SyntheticDenoiserTrainer:
    """
    Stage 2A: Supervised Pretraining on paired synthetic clean/noisy hydrophone signals.
    Minimizes MSE against ground-truth clean signals.
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or default_config
        self.cfg.validate()
        set_seed(self.cfg.RANDOM_SEED, self.cfg.DETERMINISTIC)

        dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev_str)
        logger.info("Stage 2A Trainer initialized on device: %s", self.device)

        self.model = Configurable1DUNet(cfg=self.cfg).to(self.device)

    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        num_epochs = epochs if epochs is not None else self.cfg.PRETRAIN_EPOCHS
        b_size = batch_size if batch_size is not None else self.cfg.PRETRAIN_BATCH_SIZE
        learning_rate = lr if lr is not None else self.cfg.PRETRAIN_LR
        early_patience = patience if patience is not None else self.cfg.PRETRAIN_PATIENCE

        train_ds = UNetPretrainDataset(self.cfg, split="train", augment=True)
        val_ds = UNetPretrainDataset(self.cfg, split="val", augment=False)

        if len(train_ds) == 0:
            raise RuntimeError(
                f"Synthetic clean/noisy dataset is empty in {self.cfg.SYNTHETIC_DATA_DIR}. "
                "Run generate_trial_data.py first."
            )

        train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, drop_last=False)
        has_val = len(val_ds) > 0
        val_loader = DataLoader(val_ds, batch_size=b_size, shuffle=False) if has_val else None

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.cfg.PRETRAIN_WEIGHT_DECAY,
        )
        criterion = nn.MSELoss()
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.cfg.PRETRAIN_LR_FACTOR,
            patience=self.cfg.PRETRAIN_LR_PATIENCE,
            min_lr=self.cfg.PRETRAIN_LR_MIN,
        )

        best_loss = float("inf")
        patience_counter = 0

        logger.info("Starting Stage 2A Supervised Pretraining: %d epochs, batch_size=%d, lr=%.2e", num_epochs, b_size, learning_rate)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_loss = 0.0
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                optimizer.zero_grad()
                pred = self.model(noisy)
                loss = criterion(pred, clean)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(noisy)
            train_loss /= len(train_ds)

            # Evaluation
            if has_val and val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                noisy_baseline_loss = 0.0
                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy, clean = noisy.to(self.device), clean.to(self.device)
                        pred = self.model(noisy)
                        val_loss += criterion(pred, clean).item() * len(noisy)
                        noisy_baseline_loss += criterion(noisy, clean).item() * len(noisy)
                val_loss /= len(val_ds)
                noisy_baseline_loss /= len(val_ds)
                scheduler.step(val_loss)
                monitored = val_loss
                logger.info(
                    "Epoch %02d/%02d - Train MSE: %.5f, Val MSE: %.5f (Input Noisy MSE: %.5f)",
                    epoch, num_epochs, train_loss, val_loss, noisy_baseline_loss
                )
            else:
                monitored = train_loss
                scheduler.step(train_loss)
                logger.info("Epoch %02d/%02d - Train MSE: %.5f", epoch, num_epochs, train_loss)

            if monitored < best_loss:
                best_loss = monitored
                patience_counter = 0
                if self.cfg.PRETRAIN_SAVE_BEST:
                    ensure_dir(self.cfg.UNET_PRETRAINED_CHECKPOINT)
                    ckpt = {
                        "tag": "pretrained",
                        "epoch": epoch,
                        "val_loss": best_loss,
                        "model_state_dict": self.model.state_dict(),
                        "unet_config": self.cfg.UNET_CONFIG,
                        "signal_length": self.cfg.SIGNAL_LENGTH,
                    }
                    torch.save(ckpt, self.cfg.UNET_PRETRAINED_CHECKPOINT)
                    logger.info("Saved best pretrained model to %s (MSE: %.5f)", self.cfg.UNET_PRETRAINED_CHECKPOINT, best_loss)
            else:
                patience_counter += 1
                if patience_counter >= early_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        # Save final checkpoint
        final_ckpt_path = self.cfg.UNET_PRETRAINED_CHECKPOINT.replace(".pth", "_final.pth")
        ensure_dir(final_ckpt_path)
        torch.save({
            "tag": "pretrained_final",
            "epoch": epoch,
            "val_loss": best_loss,
            "model_state_dict": self.model.state_dict(),
            "unet_config": self.cfg.UNET_CONFIG,
            "signal_length": self.cfg.SIGNAL_LENGTH,
        }, final_ckpt_path)
        logger.info("Saved final pretrained model to %s", final_ckpt_path)

        # Print mandatory manual verification notice
        print("\n" + "=" * 70)
        print("STAGE 2A PRETRAINING COMPLETE.")
        print("ATTENTION: MANUAL VERIFICATION IS REQUIRED BEFORE FINE-TUNING!")
        print("Inspect inference outputs and SNR improvements on synthetic validation data.")
        print(f"Pretrained checkpoint: {self.cfg.UNET_PRETRAINED_CHECKPOINT}")
        print("=" * 70 + "\n")

        return {"best_loss": best_loss}


class TrialDataFineTuner:
    """
    Stage 2B: Unsupervised Fine-Tuning on real hydrophone trial recordings.
    Loads pretrained Stage 2A checkpoint, verifies architecture compatibility,
    and optimizes UnsupervisedNoiseSuppressionLoss against quiet recording noise references.
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or default_config
        self.cfg.validate()
        set_seed(self.cfg.RANDOM_SEED, self.cfg.DETERMINISTIC)

        dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev_str)

        # Refuse to run unless pretrained checkpoint exists
        if not os.path.isfile(self.cfg.UNET_PRETRAINED_CHECKPOINT):
            raise FileNotFoundError(
                f"Stage 2A pretrained checkpoint not found at: '{self.cfg.UNET_PRETRAINED_CHECKPOINT}'. "
                "You must complete Stage 2A pretraining and manual verification before Stage 2B fine-tuning."
            )

        self.model = Configurable1DUNet(cfg=self.cfg).to(self.device)
        self._load_checkpoint(self.cfg.UNET_PRETRAINED_CHECKPOINT)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Validate checkpoint configuration and load weights.
        """
        logger.info("Loading pretrained checkpoint from %s", checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Validate stored architecture matches current config
        stored_features = ckpt.get("unet_config", {}).get("features")
        cfg_features = self.cfg.UNET_CONFIG.get("features")
        if stored_features != cfg_features:
            raise ValueError(
                f"UNet feature mismatch! Checkpoint features: {stored_features}, Config features: {cfg_features}"
            )

        stored_sig_len = ckpt.get("signal_length")
        if stored_sig_len != self.cfg.TRIAL_SIGNAL_LENGTH:
            logger.warning(
                "Stored signal length (%s) differs from TRIAL_SIGNAL_LENGTH (%d). "
                "Network is fully-convolutional and handles variable lengths.",
                stored_sig_len, self.cfg.TRIAL_SIGNAL_LENGTH
            )

        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Successfully loaded Stage 2A weights (Epoch %s, Loss %s)", ckpt.get("epoch"), ckpt.get("val_loss"))

    def finetune(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        patience: Optional[int] = None,
    ) -> Dict[str, Any]:
        num_epochs = epochs if epochs is not None else self.cfg.FINETUNE_EPOCHS
        b_size = batch_size if batch_size is not None else self.cfg.FINETUNE_BATCH_SIZE
        learning_rate = lr if lr is not None else self.cfg.FINETUNE_LR
        early_patience = patience if patience is not None else self.cfg.FINETUNE_PATIENCE

        train_ds = NoiseAdaptationUNetDataset(self.cfg, subset="all")
        if len(train_ds) == 0:
            raise RuntimeError(
                f"No trial data available in {self.cfg.TRIAL_DATA_DIR}. "
                "Run generate_trial_data.py first."
            )

        train_loader = DataLoader(train_ds, batch_size=b_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.cfg.FINETUNE_WEIGHT_DECAY,
        )
        criterion = UnsupervisedNoiseSuppressionLoss(
            lam=self.cfg.FINETUNE_NOISE_LAMBDA,
            smooth_scale=getattr(self.cfg, "FINETUNE_SMOOTH_SCALE", 0.05),
            phase_weight=getattr(self.cfg, "FINETUNE_PHASE_WEIGHT", 0.05),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.cfg.FINETUNE_LR_FACTOR,
            patience=self.cfg.FINETUNE_LR_PATIENCE,
            min_lr=self.cfg.FINETUNE_LR_MIN,
        )

        best_loss = float("inf")
        patience_counter = 0

        logger.info("Starting Stage 2B Fine-Tuning: %d epochs, batch_size=%d, lr=%.2e", num_epochs, b_size, learning_rate)

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for noisy, noise_ref in train_loader:
                noisy, noise_ref = noisy.to(self.device), noise_ref.to(self.device)
                optimizer.zero_grad()
                denoised = self.model(noisy)
                loss = criterion(denoised, noisy, noise_ref)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(noisy)
            epoch_loss /= len(train_ds)
            scheduler.step(epoch_loss)

            logger.info("Epoch %02d/%02d - Unsupervised Loss: %.5f", epoch, num_epochs, epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0

                # Ensure we save to separate checkpoint and NEVER overwrite the pretrained checkpoint
                assert self.cfg.UNET_FINETUNED_CHECKPOINT != self.cfg.UNET_PRETRAINED_CHECKPOINT, (
                    "Checkpoint collision: UNET_FINETUNED_CHECKPOINT must differ from UNET_PRETRAINED_CHECKPOINT"
                )

                ensure_dir(self.cfg.UNET_FINETUNED_CHECKPOINT)
                ckpt = {
                    "tag": "finetuned",
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "model_state_dict": self.model.state_dict(),
                    "unet_config": self.cfg.UNET_CONFIG,
                    "signal_length": self.cfg.TRIAL_SIGNAL_LENGTH,
                }
                torch.save(ckpt, self.cfg.UNET_FINETUNED_CHECKPOINT)
                logger.info("Saved fine-tuned checkpoint to %s (Loss: %.5f)", self.cfg.UNET_FINETUNED_CHECKPOINT, best_loss)
            else:
                patience_counter += 1
                if patience_counter >= early_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        logger.info("Stage 2B Fine-Tuning complete. Best Loss: %.5f", best_loss)
        return {"best_loss": best_loss}


def main():
    parser = argparse.ArgumentParser(description="Stage 2 U-Net Denoiser (Pretrain / Finetune)")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "finetune"],
        default=None,
        help="Stage to execute: 'pretrain' (Stage 2A) or 'finetune' (Stage 2B). Defaults to 'pretrain'.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    args = parser.parse_args()

    stage = args.stage
    if stage is None:
        logger.warning("No --stage argument supplied. Defaulting to 'pretrain'.")
        stage = "pretrain"

    cfg = Config()

    if stage == "pretrain":
        trainer = SyntheticDenoiserTrainer(cfg=cfg, device=args.device)
        trainer.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif stage == "finetune":
        print("\n" + "=" * 70)
        print("STAGE 2B: UNSUPERVISED NOISE ADAPTATION FINE-TUNING")
        print("Reminder: Ensure Stage 2A manual verification was completed successfully.")
        print("Waiting 3 seconds before commencing fine-tuning...")
        print("=" * 70 + "\n")
        time.sleep(3)

        finetuner = TrialDataFineTuner(cfg=cfg, device=args.device)
        finetuner.finetune(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()

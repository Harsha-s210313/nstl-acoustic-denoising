"""
Configuration module for underwater acoustic denoising pipeline.
Single source of truth for all parameters, hyperparameters, and relative file paths.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os


@dataclass
class Config:
    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    RANDOM_SEED: int = 42
    DETERMINISTIC: bool = False

    # --------------------------------------------------------------------------
    # Synthetic Data Parameters (100 kHz)
    # --------------------------------------------------------------------------
    SIGNAL_LENGTH: int = 25000  # Model input length for synthetic stage
    SAMPLE_RATE: float = 100_000.0  # Hz (4 s recordings, 25000 sample model window)
    SYNTHETIC_IS_COMPLEX: bool = False

    # --------------------------------------------------------------------------
    # Extraction Parameters (Synthetic 100 kHz)
    # --------------------------------------------------------------------------
    EXTRACTION_RMS_WINDOW: int = 1000
    EXTRACTION_THRESHOLD_FACTOR: float = 2.0
    EXTRACTION_PAD_SAMPLES: int = 5000
    EXTRACTION_NOISE_PERCENTILE: float = 20.0
    NOISE_REF_WINDOW: int = 2000

    # --------------------------------------------------------------------------
    # Trial Data Parameters (17.8 kHz bandpass-sampled)
    # --------------------------------------------------------------------------
    TRIAL_SAMPLE_RATE: float = 17_800.0  # Hz
    TRIAL_SIGNAL_LENGTH: int = 5000  # Model input length for trial stage
    TRIAL_RMS_WINDOW: int = 178  # 10 ms at 17.8 kHz
    TRIAL_PAD_SAMPLES: int = 890  # 50 ms at 17.8 kHz
    TRIAL_NOISE_REF_WINDOW: int = 356  # 20 ms at 17.8 kHz
    TRIAL_IS_COMPLEX: bool = False

    # --------------------------------------------------------------------------
    # Directory & Checkpoint Paths
    # --------------------------------------------------------------------------
    SYNTHETIC_DATA_DIR: str = r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\data"
    SYNTHETIC_CLEAN_SUBDIR: str = "synthetic/clean"
    SYNTHETIC_NOISY_SUBDIR: str = "synthetic/noisy"

    TRIAL_DATA_DIR: str = r"C:\Users\HARSHA\Documents\pri_generated"
    TRIAL_SIGNAL_SUBDIR: str = ""
    TRIAL_NOISE_SUBDIR: str = ""  # Optional, may not exist

    MODELS_DIR: str = "models"
    UNET_PRETRAINED_CHECKPOINT: str = "models/unet_pretrained_best.pth"
    UNET_FINETUNED_CHECKPOINT: str = "models/unet_finetuned_best.pth"

    INFERENCE_OUTPUT_DIR: str = "inference_output"
    INFERENCE_DEFAULT_SOURCE: str = "trial"
    INFERENCE_TRIAL_PRI: str = ""
    INFERENCE_TRIAL_CHANNEL: str = "channel24.dat"
    INFERENCE_RECONSTRUCT_FULL: bool = True
    INFERENCE_FULL_SEQUENCE: bool = True
    INFERENCE_CLEANUP_BPF: bool = True

    # --------------------------------------------------------------------------
    # U-Net Architecture Configuration
    # --------------------------------------------------------------------------
    UNET_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        "features": [16, 32, 64, 128],
        "kernel_size": 7,
        "dropout": 0.1,
        "in_channels": 1,
        "out_channels": 1,
    })

    # --------------------------------------------------------------------------
    # Stage 2A: Pretraining Hyperparameters (Supervised Synthetic Clean/Noisy)
    # --------------------------------------------------------------------------
    PRETRAIN_EPOCHS: int = 100
    PRETRAIN_BATCH_SIZE: int = 8
    PRETRAIN_LR: float = 1e-3
    PRETRAIN_WEIGHT_DECAY: float = 1e-4
    PRETRAIN_VAL_SPLIT: float = 0.15
    PRETRAIN_PATIENCE: int = 20
    PRETRAIN_SAVE_BEST: bool = True
    PRETRAIN_LR_FACTOR: float = 0.5
    PRETRAIN_LR_PATIENCE: int = 10
    PRETRAIN_LR_MIN: float = 1e-6

    # Exact field names from prompt specifications
    EPOCHS: int = 100
    BATCH: int = 8
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    VAL_SPLIT: float = 0.15
    PATIENCE: int = 20
    SAVE_BEST: bool = True
    LR_FACTOR: float = 0.5
    LR_PATIENCE: int = 10
    LR_MIN: float = 1e-6

    # --------------------------------------------------------------------------
    # Data Augmentation (Pretraining)
    # --------------------------------------------------------------------------
    AUG_ENABLED: bool = True
    AUG_TIME_SHIFT_MAX: int = 50
    AUG_AMPLITUDE_JITTER: float = 0.1
    AUG_NOISE_STD_MAX: float = 0.05
    AUG_FLIP_PROB: float = 0.0

    # --------------------------------------------------------------------------
    # Stage 2B: Fine-Tuning Hyperparameters (Unsupervised Noise Adaptation)
    # --------------------------------------------------------------------------
    FINETUNE_EPOCHS: int = 30
    FINETUNE_BATCH_SIZE: int = 4
    FINETUNE_LR: float = 1e-4
    FINETUNE_WEIGHT_DECAY: float = 1e-4
    FINETUNE_PATIENCE: int = 10
    FINETUNE_LR_FACTOR: float = 0.5
    FINETUNE_LR_PATIENCE: int = 5
    FINETUNE_LR_MIN: float = 1e-7
    FINETUNE_NOISE_LAMBDA: float = 0.7
    FINETUNE_SMOOTH_SCALE: float = 0.05
    FINETUNE_PHASE_WEIGHT: float = 0.05

    def validate(self) -> None:
        """
        Validate all configuration settings, raising clear ValueError/TypeError on mismatches.
        """
        if self.SIGNAL_LENGTH <= 0:
            raise ValueError(f"SIGNAL_LENGTH must be positive, got {self.SIGNAL_LENGTH}")
        if self.TRIAL_SIGNAL_LENGTH <= 0:
            raise ValueError(f"TRIAL_SIGNAL_LENGTH must be positive, got {self.TRIAL_SIGNAL_LENGTH}")
        if self.SAMPLE_RATE <= 0:
            raise ValueError(f"SAMPLE_RATE must be positive, got {self.SAMPLE_RATE}")
        if self.TRIAL_SAMPLE_RATE <= 0:
            raise ValueError(f"TRIAL_SAMPLE_RATE must be positive, got {self.TRIAL_SAMPLE_RATE}")
        if self.TRIAL_RMS_WINDOW <= 0 or self.TRIAL_RMS_WINDOW > self.TRIAL_SIGNAL_LENGTH:
            raise ValueError(f"Invalid TRIAL_RMS_WINDOW: {self.TRIAL_RMS_WINDOW}")
        if self.TRIAL_NOISE_REF_WINDOW <= 0:
            raise ValueError(f"TRIAL_NOISE_REF_WINDOW must be positive, got {self.TRIAL_NOISE_REF_WINDOW}")

        # UNet config validation
        if not isinstance(self.UNET_CONFIG, dict):
            raise TypeError("UNET_CONFIG must be a dictionary")
        required_unet_keys = ["features", "kernel_size", "dropout", "in_channels", "out_channels"]
        for k in required_unet_keys:
            if k not in self.UNET_CONFIG:
                raise ValueError(f"Missing required key in UNET_CONFIG: '{k}'")

        features = self.UNET_CONFIG["features"]
        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise ValueError(f"UNET_CONFIG['features'] must be a non-empty list of integers, got {features}")
        for f in features:
            if not isinstance(f, int) or f <= 0:
                raise ValueError(f"UNET_CONFIG feature widths must be positive ints, got {f}")

        k = self.UNET_CONFIG["kernel_size"]
        if not isinstance(k, int) or k <= 0 or k % 2 == 0:
            raise ValueError(f"UNET_CONFIG['kernel_size'] must be a positive odd integer, got {k}")

        # Splits and Lambda validation
        if not (0.0 <= self.PRETRAIN_VAL_SPLIT < 1.0):
            raise ValueError(f"PRETRAIN_VAL_SPLIT must be in [0, 1), got {self.PRETRAIN_VAL_SPLIT}")
        if not (0.0 <= self.FINETUNE_NOISE_LAMBDA <= 1.0):
            raise ValueError(f"FINETUNE_NOISE_LAMBDA must be in [0, 1], got {self.FINETUNE_NOISE_LAMBDA}")
        if not (0.0 <= self.FINETUNE_SMOOTH_SCALE <= 1.0):
            raise ValueError(f"FINETUNE_SMOOTH_SCALE must be in [0, 1], got {self.FINETUNE_SMOOTH_SCALE}")
        if not (0.0 <= self.FINETUNE_PHASE_WEIGHT <= 1.0):
            raise ValueError(f"FINETUNE_PHASE_WEIGHT must be in [0, 1], got {self.FINETUNE_PHASE_WEIGHT}")


# Default singleton instance
default_config = Config()

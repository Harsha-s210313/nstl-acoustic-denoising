
import os

class Config:
    RANDOM_SEED: int = 42
    DETERMINISTIC: bool = False

    # ------------------------------------------------------------------ #
    # Signal window length
    # At 100 kHz:  25000 samples = 250 ms
    #   → covers 100ms LFM pulse + 60ms reverb + 90ms margin on both sides
    # NOTE: The pretrained U-Net is fully convolutional — changing this
    # value does NOT invalidate pretrained weights. Fine-tuning adapts them.
    # ------------------------------------------------------------------ #
    SIGNAL_LENGTH: int = 25000
    SAMPLE_RATE: float = 100_000.0    # Hz

    # ------------------------------------------------------------------ #
    # Active-region extraction  (signal_extractor + trial dataset)
    # At 100 kHz:
    #   EXTRACTION_RMS_WINDOW = 1000  →  10 ms smoothing window
    #   EXTRACTION_PAD_SAMPLES = 5000 →  50 ms margin on each side
    # Tune EXTRACTION_THRESHOLD_FACTOR if:
    #   too high → signal edges get clipped, or nothing detected
    #   too low  → noise spikes get included in the active region
    # ------------------------------------------------------------------ #
    EXTRACTION_RMS_WINDOW: int        = 1000
    EXTRACTION_THRESHOLD_FACTOR: float = 2.0   # works for sine-wave LFM (RMS = amp/√2)
    EXTRACTION_PAD_SAMPLES: int       = 5000   # 50 ms reverb margin
    EXTRACTION_NOISE_PERCENTILE: float = 20.0

    SYNTHETIC_IS_COMPLEX: bool = False
    TRIAL_IS_COMPLEX: bool = False

    SYNTHETIC_DATA_DIR: str = os.path.join("data", "synthetic")
    SYNTHETIC_CLEAN_SUBDIR: str = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Synthetic_Data\clean"
    SYNTHETIC_NOISY_SUBDIR: str = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Synthetic_Data\noisy"

    TRIAL_DATA_DIR: str = os.path.join("data", "trial_data")
    # _scan_pri_tree expects:  <SUBDIR>/PRI_XX/channelYYY.dat
    # So point one level ABOVE PRI_TEST → scanner finds PRI_TEST as the PRI folder.
    TRIAL_SIGNAL_SUBDIR: str = r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\data\trial_data\Signal"
    # Noise folder doesn't exist for generated data → skipped with a warning (noise
    # reference is extracted from inside each signal file automatically).
    TRIAL_NOISE_SUBDIR: str  = r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\data\trial_data\Noise"

    DETECTOR_DATA_DIR: str = os.path.join("data", "detector")

    MODELS_DIR: str = "models"
    DETECTOR_CHECKPOINT: str = os.path.join("models", "detector.pth")
    UNET_PRETRAINED_CHECKPOINT: str = os.path.join("models", "unet_pretrained.pth")
    UNET_FINETUNED_CHECKPOINT: str = os.path.join("models", "unet_finetuned.pth")

    UNET_CONFIG: dict = {
        "features": [16, 32, 64, 128],
        "kernel_size": 7,
        "dropout": 0.1,
        "in_channels": 1,
        "out_channels": 1,
    }

    DETECTOR_FEATURES: list = [16, 32, 64]
    DETECTOR_KERNEL_SIZE: int = 7
    DETECTOR_DROPOUT: float = 0.3
    DETECTOR_FC_HIDDEN: int = 128

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

    AUG_ENABLED: bool = True
    AUG_TIME_SHIFT_MAX: int = 50
    AUG_AMPLITUDE_JITTER: float = 0.1
    AUG_NOISE_STD_MAX: float = 0.05
    AUG_FLIP_PROB: float = 0.0

    FINETUNE_EPOCHS: int = 30
    FINETUNE_BATCH_SIZE: int = 4
    FINETUNE_LR: float = 1e-4
    FINETUNE_WEIGHT_DECAY: float = 1e-4
    FINETUNE_PATIENCE: int = 10
    FINETUNE_LR_FACTOR: float = 0.5
    FINETUNE_LR_PATIENCE: int = 5
    FINETUNE_LR_MIN: float = 1e-7
    FINETUNE_NOISE_LAMBDA: float = 1.0
    NOISE_REF_WINDOW: int = 2000   # 20 ms at 100 kHz

    DETECTOR_EPOCHS: int = 50
    DETECTOR_BATCH_SIZE: int = 16
    DETECTOR_LR: float = 1e-3
    DETECTOR_WEIGHT_DECAY: float = 1e-4
    DETECTOR_VAL_SPLIT: float = 0.2
    DETECTOR_PATIENCE: int = 15

    INFERENCE_DEFAULT_SOURCE: str = "synthetic"
    INFERENCE_TRIAL_PRI: str = "pri01"
    INFERENCE_TRIAL_CHANNEL: int = 0
    INFERENCE_OUTPUT_DIR: str = "inference_output"

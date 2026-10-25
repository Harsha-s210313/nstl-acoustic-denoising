"""
NSTL Underwater Acoustic Signal Denoising Package.
"""

from .config import Config, default_config
from .utils import (
    set_seed,
    load_dat_file,
    save_dat_file,
    compute_rms_envelope,
    estimate_noise_floor,
    detect_active_region,
    extract_active_region,
    extract_noise_reference,
    normalize_signal,
    denormalize_signal,
    to_tensor,
    ensure_dir,
)
from .unet import Configurable1DUNet
from .denoiser_finetune import (
    UnsupervisedNoiseSuppressionLoss,
    SyntheticDenoiserTrainer,
    TrialDataFineTuner,
)

__all__ = [
    "Config",
    "default_config",
    "set_seed",
    "load_dat_file",
    "save_dat_file",
    "compute_rms_envelope",
    "estimate_noise_floor",
    "detect_active_region",
    "extract_active_region",
    "extract_noise_reference",
    "normalize_signal",
    "denormalize_signal",
    "to_tensor",
    "ensure_dir",
    "Configurable1DUNet",
    "UnsupervisedNoiseSuppressionLoss",
    "SyntheticDenoiserTrainer",
    "TrialDataFineTuner",
]

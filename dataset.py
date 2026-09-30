"""
PyTorch Dataset implementations for:
1. DetectorDataset: Supervised pulse detection (Signal=1, Noise=0).
2. UNetPretrainDataset: Supervised pretraining on matched synthetic clean/noisy pairs.
3. NoiseAdaptationUNetDataset: Unsupervised noise adaptation on trial data.
"""

import os
import glob
import logging
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .config import Config
    from .utils import (
        load_dat_file,
        normalize_signal,
        compute_rms_envelope,
        estimate_noise_floor,
        detect_active_region,
        extract_noise_reference,
        to_tensor,
    )
except (ImportError, ValueError):
    from config import Config
    from utils import (
        load_dat_file,
        normalize_signal,
        compute_rms_envelope,
        estimate_noise_floor,
        detect_active_region,
        extract_noise_reference,
        to_tensor,
    )

logger = logging.getLogger(__name__)


class DetectorDataset(Dataset):
    """
    Dataset for Stage 1 CNN Detector.
    Loads signals from DETECTOR_DATA_DIR/signal (label 1) and DETECTOR_DATA_DIR/noise (label 0).
    Applies independent z-score normalization and optional training jitter.
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.augment = augment and (split == "train") and cfg.AUG_ENABLED

        sig_dir = os.path.join(cfg.DETECTOR_DATA_DIR, "signal")
        if not os.path.isdir(sig_dir):
            for alt in [
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "detector", "signal"),
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "signal"),
            ]:
                if os.path.isdir(alt):
                    sig_dir = alt
                    break

        noise_dir = os.path.join(cfg.DETECTOR_DATA_DIR, "noise")
        if not os.path.isdir(noise_dir):
            for alt in [
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "detector", "noise"),
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "noise"),
            ]:
                if os.path.isdir(alt):
                    noise_dir = alt
                    break

        samples: List[Tuple[str, int]] = []
        if os.path.isdir(sig_dir):
            for f in sorted(glob.glob(os.path.join(sig_dir, "*.dat"))):
                samples.append((f, 1))
        if os.path.isdir(noise_dir):
            for f in sorted(glob.glob(os.path.join(noise_dir, "*.dat"))):
                samples.append((f, 0))

        if not samples:
            logger.warning("No detector data found in %s", cfg.DETECTOR_DATA_DIR)

        # Deterministic shuffle & split
        rng = np.random.RandomState(cfg.RANDOM_SEED)
        indices = np.arange(len(samples))
        rng.shuffle(indices)

        val_size = int(len(samples) * cfg.DETECTOR_VAL_SPLIT)
        if split == "train":
            chosen_indices = indices[val_size:]
        else:
            chosen_indices = indices[:val_size]

        self.samples = [samples[i] for i in chosen_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        data = load_dat_file(path, expected_length=self.cfg.SIGNAL_LENGTH)
        norm_data, _, _ = normalize_signal(data)

        if self.augment and self.cfg.AUG_AMPLITUDE_JITTER > 0:
            jitter = np.random.uniform(
                1.0 - self.cfg.AUG_AMPLITUDE_JITTER,
                1.0 + self.cfg.AUG_AMPLITUDE_JITTER,
            )
            norm_data = norm_data * jitter

        x = to_tensor(norm_data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class UNetPretrainDataset(Dataset):
    """
    Dataset for Stage 2A Supervised U-Net pretraining.
    Pairs clean and noisy recordings by identical filename.
    Applies independent z-score normalization and on-the-fly pair augmentations.
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.augment = augment and (split == "train") and cfg.AUG_ENABLED

        clean_dir = os.path.join(cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_CLEAN_SUBDIR)
        if not os.path.isdir(clean_dir):
            for alt in [
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "synthetic", "clean"),
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "clean"),
            ]:
                if os.path.isdir(alt):
                    clean_dir = alt
                    break

        noisy_dir = os.path.join(cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_NOISY_SUBDIR)
        if not os.path.isdir(noisy_dir):
            for alt in [
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "synthetic", "noisy"),
                os.path.join(cfg.SYNTHETIC_DATA_DIR, "noisy"),
            ]:
                if os.path.isdir(alt):
                    noisy_dir = alt
                    break

        clean_files = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(clean_dir, "*.dat"))
        }
        noisy_files = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(noisy_dir, "*.dat"))
        }

        common_keys = sorted(list(set(clean_files.keys()) & set(noisy_files.keys())))
        unmatched_clean = set(clean_files.keys()) - set(noisy_files.keys())
        unmatched_noisy = set(noisy_files.keys()) - set(clean_files.keys())

        if unmatched_clean:
            logger.warning("UNetPretrainDataset: %d unmatched clean files", len(unmatched_clean))
        if unmatched_noisy:
            logger.warning("UNetPretrainDataset: %d unmatched noisy files", len(unmatched_noisy))

        pairs = [(noisy_files[k], clean_files[k]) for k in common_keys]

        # Deterministic shuffle & split
        rng = np.random.RandomState(cfg.RANDOM_SEED)
        indices = np.arange(len(pairs))
        rng.shuffle(indices)

        val_size = int(len(pairs) * cfg.PRETRAIN_VAL_SPLIT)
        if split == "train":
            chosen_indices = indices[val_size:]
        else:
            chosen_indices = indices[:val_size]

        self.pairs = [pairs[i] for i in chosen_indices]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.pairs[idx]

        noisy = load_dat_file(noisy_path, expected_length=self.cfg.SIGNAL_LENGTH)
        clean = load_dat_file(clean_path, expected_length=self.cfg.SIGNAL_LENGTH)

        # Independent normalization
        noisy, _, _ = normalize_signal(noisy)
        clean, _, _ = normalize_signal(clean)

        # Augmentation on-the-fly (train only)
        if self.augment:
            # 1. Circular time shift applied identically to both
            if self.cfg.AUG_TIME_SHIFT_MAX > 0:
                shift = int(np.random.randint(-self.cfg.AUG_TIME_SHIFT_MAX, self.cfg.AUG_TIME_SHIFT_MAX + 1))
                noisy = np.roll(noisy, shift)
                clean = np.roll(clean, shift)

            # 2. Amplitude scale applied identically to both
            if self.cfg.AUG_AMPLITUDE_JITTER > 0:
                scale = float(np.random.uniform(
                    1.0 - self.cfg.AUG_AMPLITUDE_JITTER,
                    1.0 + self.cfg.AUG_AMPLITUDE_JITTER,
                ))
                noisy = noisy * scale
                clean = clean * scale

            # 3. Extra additive white noise on noisy only
            if self.cfg.AUG_NOISE_STD_MAX > 0:
                noise_std = float(np.random.uniform(0.0, self.cfg.AUG_NOISE_STD_MAX))
                noisy = noisy + np.random.normal(0.0, noise_std, size=noisy.shape).astype(np.float32)

            # 4. Optional time reversal applied identically to both
            if self.cfg.AUG_FLIP_PROB > 0 and np.random.random() < self.cfg.AUG_FLIP_PROB:
                noisy = np.flip(noisy).copy()
                clean = np.flip(clean).copy()

        return to_tensor(noisy, dtype=torch.float32), to_tensor(clean, dtype=torch.float32)


class NoiseAdaptationUNetDataset(Dataset):
    """
    Dataset for Stage 2B Unsupervised Noise Adaptation on real trial recordings.
    Scans PRI/channel folders, extracts active pulse window and an inactive noise reference.
    Normalizes noise reference using the active window's mean and std.
    """

    def __init__(
        self,
        cfg: Config,
        subset: str = "all",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.subset = subset

        search_dirs = []
        if cfg.TRIAL_SIGNAL_SUBDIR:
            sig_path = (
                cfg.TRIAL_SIGNAL_SUBDIR
                if os.path.isabs(cfg.TRIAL_SIGNAL_SUBDIR)
                else os.path.join(cfg.TRIAL_DATA_DIR, cfg.TRIAL_SIGNAL_SUBDIR)
            )
            if os.path.isdir(sig_path) and subset in ("all", "signal"):
                search_dirs.append(sig_path)

        if cfg.TRIAL_NOISE_SUBDIR:
            noise_path = (
                cfg.TRIAL_NOISE_SUBDIR
                if os.path.isabs(cfg.TRIAL_NOISE_SUBDIR)
                else os.path.join(cfg.TRIAL_DATA_DIR, cfg.TRIAL_NOISE_SUBDIR)
            )
            if os.path.isdir(noise_path) and subset in ("all", "noise"):
                search_dirs.append(noise_path)

        if not search_dirs and os.path.isdir(cfg.TRIAL_DATA_DIR):
            search_dirs.append(cfg.TRIAL_DATA_DIR)

        self.files: List[str] = []
        for s_dir in search_dirs:
            # Recursive scan for PRI_xx/channel*.dat or any *.dat
            matches = glob.glob(os.path.join(s_dir, "**", "*.dat"), recursive=True)
            self.files.extend(matches)

        self.files = sorted(list(set(self.files)))
        if not self.files:
            logger.warning("No trial .dat files found across: %s", search_dirs)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.files[idx]
        raw_signal = load_dat_file(file_path)
        sig_len = len(raw_signal)

        # 1. Compute RMS envelope and noise floor
        rms = compute_rms_envelope(raw_signal, self.cfg.TRIAL_RMS_WINDOW)
        noise_floor = estimate_noise_floor(rms, self.cfg.EXTRACTION_NOISE_PERCENTILE)

        # 2. Detect active pulse region
        start, end = detect_active_region(
            rms,
            noise_floor,
            self.cfg.EXTRACTION_THRESHOLD_FACTOR,
            self.cfg.TRIAL_PAD_SAMPLES,
            sig_len,
        )

        # 3. Extract active window padded/truncated to TRIAL_SIGNAL_LENGTH
        act_slice = raw_signal[start:end]
        target_len = self.cfg.TRIAL_SIGNAL_LENGTH
        if len(act_slice) < target_len:
            active_window = np.pad(act_slice, (0, target_len - len(act_slice)), mode="constant")
        else:
            active_window = act_slice[:target_len]

        # 4. Extract TRIAL_NOISE_REF_WINDOW reference (prefer immediately before, then after, then fallback)
        w_ref = self.cfg.TRIAL_NOISE_REF_WINDOW
        if start >= w_ref:
            noise_ref = raw_signal[start - w_ref: start]
        elif end + w_ref <= sig_len:
            noise_ref = raw_signal[end: end + w_ref]
        elif sig_len >= w_ref:
            noise_ref = raw_signal[:w_ref]
        else:
            noise_ref = extract_noise_reference(raw_signal, w_ref)

        if len(noise_ref) < w_ref:
            noise_ref = np.pad(noise_ref, (0, w_ref - len(noise_ref)), mode="constant")
        else:
            noise_ref = noise_ref[:w_ref]

        # 5. Shared normalization: normalize active window, and scale noise reference with identical mean and std
        norm_active, mean, std = normalize_signal(active_window)
        norm_noise_ref = (noise_ref - mean) / (std + 1e-8)

        x_active = to_tensor(norm_active, dtype=torch.float32)
        x_noise_ref = to_tensor(norm_noise_ref, dtype=torch.float32)

        return x_active, x_noise_ref

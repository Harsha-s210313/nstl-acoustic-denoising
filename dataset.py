"""
dataset.py
----------
PyTorch Dataset implementations for the NSTL Acoustic Pipeline.

Classes
-------
DetectorDataset
    Stage 1: binary classification dataset (signal-present vs. noise-only).

UNetPretrainDataset
    Stage 2A: paired clean/noisy synthetic LFM data for supervised denoising.

NoiseAdaptationUNetDataset
    Stage 2B: real trial data with automatically extracted noise references
    for unsupervised fine-tuning.
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import Config
from utils import load_dat_file, extract_noise_reference, extract_active_region, normalize_signal, to_tensor

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _collect_dat_files(directory: str) -> List[str]:
    """
    Return a sorted list of absolute paths to all .dat files directly inside
    ``directory`` (non-recursive, first level only).

    Raises FileNotFoundError if the directory does not exist.
    Raises ValueError if the directory contains no .dat files.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory!r}. "
            "Check your Config paths and make sure the data is in place."
        )
    paths = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".dat")
    )
    if not paths:
        raise ValueError(f"No .dat files found in {directory!r}.")
    return paths


def _apply_augmentation(
    noisy: np.ndarray,
    clean: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply on-the-fly domain randomisation to a (noisy, clean) pair.

    All augmentation parameters come from ``cfg``; no magic numbers here.
    The SAME transform is applied to both noisy and clean to preserve the
    paired relationship.

    Augmentations (all individually configurable):
      1. Circular time-shift    — AUG_TIME_SHIFT_MAX
      2. Amplitude jitter       — AUG_AMPLITUDE_JITTER
      3. Additional white noise — AUG_NOISE_STD_MAX  (applied to noisy only)
      4. Time reversal          — AUG_FLIP_PROB

    Parameters
    ----------
    noisy, clean : np.ndarray
        1-D float32 arrays of equal length.
    cfg : Config
        Configuration object.
    rng : np.random.Generator
        NumPy random generator (for reproducibility).

    Returns
    -------
    (noisy_aug, clean_aug) : Tuple[np.ndarray, np.ndarray]
    """
    noisy = noisy.copy()
    clean = clean.copy()

    # 1. Circular time-shift (applied identically to both channels)
    if cfg.AUG_TIME_SHIFT_MAX > 0:
        shift = int(rng.integers(-cfg.AUG_TIME_SHIFT_MAX, cfg.AUG_TIME_SHIFT_MAX + 1))
        if shift != 0:
            noisy = np.roll(noisy, shift)
            clean = np.roll(clean, shift)

    # 2. Amplitude jitter (identical scale to both to preserve SNR relationship)
    if cfg.AUG_AMPLITUDE_JITTER > 0.0:
        scale = 1.0 + float(rng.uniform(-cfg.AUG_AMPLITUDE_JITTER, cfg.AUG_AMPLITUDE_JITTER))
        noisy *= scale
        clean *= scale

    # 3. Additional synthetic noise (applied to noisy only — degrades the input,
    #    not the target, which is the physically correct direction)
    if cfg.AUG_NOISE_STD_MAX > 0.0:
        std = float(rng.uniform(0.0, cfg.AUG_NOISE_STD_MAX))
        if std > 0.0:
            noisy = noisy + rng.normal(0.0, std, size=noisy.shape).astype(np.float32)

    # 4. Time reversal
    if cfg.AUG_FLIP_PROB > 0.0 and rng.random() < cfg.AUG_FLIP_PROB:
        noisy = noisy[::-1].copy()
        clean = clean[::-1].copy()

    return noisy, clean


# --------------------------------------------------------------------------- #
# Stage 1: Detector dataset
# --------------------------------------------------------------------------- #

class DetectorDataset(Dataset):
    """
    Binary classification dataset for Stage 1 (SignalDetectorCNN).

    Directory layout::

        DETECTOR_DATA_DIR/
            signal/   *.dat   (label = 1)
            noise/    *.dat   (label = 0)

    Each .dat file is loaded, length-validated, and z-score normalised.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    split : str
        One of "train" or "val".  The split is performed deterministically
        using Config.RANDOM_SEED.
    augment : bool
        If True (train split only), apply light amplitude jitter.
    """

    def __init__(self, cfg: Config, split: str = "train", augment: bool = False):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.cfg = cfg
        self.split = split
        self.augment = augment and split == "train"
        self.rng = np.random.default_rng(cfg.RANDOM_SEED)

        signal_dir = os.path.join(cfg.DETECTOR_DATA_DIR, "signal")
        noise_dir = os.path.join(cfg.DETECTOR_DATA_DIR, "noise")

        signal_files = _collect_dat_files(signal_dir)
        noise_files = _collect_dat_files(noise_dir)

        all_paths = [(p, 1) for p in signal_files] + [(p, 0) for p in noise_files]

        # Deterministic shuffle then split
        rng_split = np.random.default_rng(cfg.RANDOM_SEED)
        indices = rng_split.permutation(len(all_paths)).tolist()
        split_idx = int(len(indices) * (1.0 - cfg.DETECTOR_VAL_SPLIT))
        if split == "train":
            chosen = indices[:split_idx]
        else:
            chosen = indices[split_idx:]

        self.samples: List[Tuple[str, int]] = [all_paths[i] for i in chosen]
        log.info("DetectorDataset[%s]: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = load_dat_file(path, expected_length=self.cfg.SIGNAL_LENGTH)
        arr, _, _ = normalize_signal(arr)

        if self.augment and self.cfg.AUG_AMPLITUDE_JITTER > 0.0:
            scale = 1.0 + float(
                self.rng.uniform(-self.cfg.AUG_AMPLITUDE_JITTER, self.cfg.AUG_AMPLITUDE_JITTER)
            )
            arr = arr * scale

        x = to_tensor(arr)                             # (1, SIGNAL_LENGTH)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# --------------------------------------------------------------------------- #
# Stage 2A: Synthetic pretraining dataset
# --------------------------------------------------------------------------- #

class UNetPretrainDataset(Dataset):
    """
    Paired clean/noisy dataset for Stage 2A supervised U-Net pretraining.

    Directory layout (configured in Config)::

        SYNTHETIC_DATA_DIR/
            clean/  sig_001.dat  sig_002.dat  ...
            noisy/  sig_001.dat  sig_002.dat  ...

    Files are matched by IDENTICAL FILENAME across the two sub-directories.

    Each sample is z-score normalised INDEPENDENTLY (i.e., the clean and
    noisy signals are each normalised with their own statistics).  During
    real inference the normalisation must therefore be applied in the same
    per-sample manner.

    On-the-fly data augmentation is applied during training if
    ``Config.AUG_ENABLED`` is True.  All augmentation parameters come from
    Config; none are hard-coded here.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    split : str
        "train" or "val".
    augment : bool
        If True, apply on-the-fly augmentation (only effective during training).
    """

    def __init__(self, cfg: Config, split: str = "train", augment: bool = False):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.cfg = cfg
        self.split = split
        self.augment = augment and split == "train" and cfg.AUG_ENABLED
        self.rng = np.random.default_rng(cfg.RANDOM_SEED)

        clean_dir = os.path.join(cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_CLEAN_SUBDIR)
        noisy_dir = os.path.join(cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_NOISY_SUBDIR)

        # Build matched pairs by filename intersection
        clean_files = {os.path.basename(p): p for p in _collect_dat_files(clean_dir)}
        noisy_files = {os.path.basename(p): p for p in _collect_dat_files(noisy_dir)}

        common = sorted(set(clean_files.keys()) & set(noisy_files.keys()))
        if not common:
            raise ValueError(
                f"No filename overlap between {clean_dir!r} and {noisy_dir!r}. "
                "Check that clean/ and noisy/ contain identically named .dat files."
            )

        only_clean = set(clean_files.keys()) - set(noisy_files.keys())
        only_noisy = set(noisy_files.keys()) - set(clean_files.keys())
        if only_clean:
            log.warning(
                "%d clean files have no matching noisy counterpart (skipped): %s",
                len(only_clean), sorted(only_clean)[:5],
            )
        if only_noisy:
            log.warning(
                "%d noisy files have no matching clean counterpart (skipped): %s",
                len(only_noisy), sorted(only_noisy)[:5],
            )

        all_pairs = [(clean_files[f], noisy_files[f]) for f in common]

        # Deterministic split
        rng_split = np.random.default_rng(cfg.RANDOM_SEED)
        indices = rng_split.permutation(len(all_pairs)).tolist()
        split_idx = max(1, int(len(indices) * (1.0 - cfg.PRETRAIN_VAL_SPLIT)))
        if split == "train":
            chosen = indices[:split_idx]
        else:
            chosen = indices[split_idx:]
            if not chosen:
                log.warning(
                    "Validation split produced 0 samples (only %d pairs total). "
                    "Increase your dataset or reduce PRETRAIN_VAL_SPLIT.",
                    len(all_pairs),
                )

        self.pairs: List[Tuple[str, str]] = [all_pairs[i] for i in chosen]
        log.info(
            "UNetPretrainDataset[%s]: %d pairs (AUG=%s)",
            split, len(self.pairs), self.augment,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        clean_path, noisy_path = self.pairs[idx]

        clean = load_dat_file(clean_path, expected_length=self.cfg.SIGNAL_LENGTH)
        noisy = load_dat_file(noisy_path, expected_length=self.cfg.SIGNAL_LENGTH)

        # Z-score normalise each signal independently
        clean, _, _ = normalize_signal(clean)
        noisy, _, _ = normalize_signal(noisy)

        if self.augment:
            noisy, clean = _apply_augmentation(noisy, clean, self.cfg, self.rng)

        noisy_t = to_tensor(noisy)   # (1, SIGNAL_LENGTH)
        clean_t = to_tensor(clean)   # (1, SIGNAL_LENGTH)
        return noisy_t, clean_t


# --------------------------------------------------------------------------- #
# Stage 2B: Trial fine-tuning dataset
# --------------------------------------------------------------------------- #

class NoiseAdaptationUNetDataset(Dataset):
    """
    Real trial data loader for Stage 2B unsupervised U-Net fine-tuning.

    There is NO paired clean ground truth.  The dataset returns:
        (noisy_signal, noise_profile)

    where ``noise_profile`` is estimated from the signal itself by finding
    the lowest-energy window of length ``Config.NOISE_REF_WINDOW``.  This
    profile is consumed by ``UnsupervisedNoiseSuppressionLoss``.

    Directory layout::

        TRIAL_DATA_DIR/
            signal/
                pri01/  channel0.dat  channel1.dat  ...
                pri02/  channel0.dat  ...
            noise/
                pri04/  channel0.dat  ...

    Both signal/ and noise/ files are used as training data (the model
    should suppress noise in all of them, whether or not a signal is truly
    present).  PRIs are exclusive across the two subdirs.

    Parameters
    ----------
    cfg : Config
        Configuration object.
    subset : str
        One of "signal", "noise", or "all" (default).
        Controls which sub-directory tree is loaded.
    """

    def __init__(self, cfg: Config, subset: str = "all"):
        if subset not in ("signal", "noise", "all"):
            raise ValueError(f"subset must be 'signal', 'noise', or 'all', got {subset!r}")

        self.cfg = cfg
        self.subset = subset
        self.files: List[str] = []

        signal_root = os.path.join(cfg.TRIAL_DATA_DIR, cfg.TRIAL_SIGNAL_SUBDIR) \
            if not os.path.isabs(cfg.TRIAL_SIGNAL_SUBDIR) \
            else cfg.TRIAL_SIGNAL_SUBDIR

        noise_root = os.path.join(cfg.TRIAL_DATA_DIR, cfg.TRIAL_NOISE_SUBDIR) \
            if not os.path.isabs(cfg.TRIAL_NOISE_SUBDIR) \
            else cfg.TRIAL_NOISE_SUBDIR

        if subset in ("signal", "all"):
            self.files.extend(self._scan_pri_tree(signal_root))
        if subset in ("noise", "all"):
            # Noise folder is optional — noise reference is now extracted
            # from inside each signal file via extract_active_region().
            if os.path.isdir(noise_root):
                self.files.extend(self._scan_pri_tree(noise_root))
            else:
                log.warning(
                    "Noise directory not found: %r — skipping. "
                    "Noise reference will be extracted from signal files.",
                    noise_root,
                )

        if not self.files:
            raise ValueError(
                f"NoiseAdaptationUNetDataset: No .dat files found under "
                f"{cfg.TRIAL_DATA_DIR!r} (subset={subset!r}). "
                "Populate the trial data directories before fine-tuning."
            )

        log.info(
            "NoiseAdaptationUNetDataset[%s]: %d files",
            subset, len(self.files),
        )

    @staticmethod
    def _scan_pri_tree(root: str) -> List[str]:
        """
        Recursively collect all .dat files under ``root``.

        Expected structure: root/priXX/channelY.dat
        """
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Trial data directory not found: {root!r}. "
                "Check Config.TRIAL_DATA_DIR and its sub-directories."
            )
        collected = []
        for pri in sorted(os.listdir(root)):
            pri_path = os.path.join(root, pri)
            if not os.path.isdir(pri_path):
                continue
            for fname in sorted(os.listdir(pri_path)):
                if fname.lower().endswith(".dat"):
                    collected.append(os.path.join(pri_path, fname))
        return collected

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]

        # Load full trial recording at actual file length (no truncation warnings).
        signal = load_dat_file(path)

        # ------------------------------------------------------------------ #
        # Active-region extraction
        #
        # Uses energy thresholding to split the recording into:
        #   active  →  signal+noise window  →  fed to the U-Net denoiser
        #   noise   →  actual silent segments before/after the pulse
        #              →  used as noise reference in UnsupervisedNoiseSuppressionLoss
        #
        # This is physically correct: the noise reference is real recorded
        # silence from the same channel, not an estimate.
        # ------------------------------------------------------------------ #
        result = extract_active_region(signal, self.cfg)

        active    = result["active"]    # shape: (SIGNAL_LENGTH,)
        noise_raw = result["noise"]     # shape: (variable — actual noise region)

        # Z-score normalise the active window
        active, _, _ = normalize_signal(active)

        # Take a fixed-length noise reference from the real noise region
        W = self.cfg.NOISE_REF_WINDOW
        if len(noise_raw) >= W:
            # Use the middle of the noise region (most representative)
            mid = len(noise_raw) // 2
            noise_ref = noise_raw[mid - W // 2 : mid - W // 2 + W]
        else:
            noise_ref = np.pad(noise_raw, (0, W - len(noise_raw)), mode="constant")

        noise_ref, _, _ = normalize_signal(noise_ref.astype(np.float32))

        signal_t = to_tensor(active)      # (1, SIGNAL_LENGTH)
        noise_t  = to_tensor(noise_ref)   # (1, NOISE_REF_WINDOW)
        return signal_t, noise_t


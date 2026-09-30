"""
Stateless, import-safe utility functions for signal processing, file I/O,
envelope detection, and tensor conversions.
"""

import os
import random
import logging
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import torch

try:
    from .config import Config
except (ImportError, ValueError):
    from config import Config

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set seeds across Python, NumPy, and PyTorch for reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists on disk.
    If path has a file extension or is a file path, creates its parent directory.
    """
    if os.path.splitext(path)[1]:
        dir_path = os.path.dirname(path)
    else:
        dir_path = path
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def load_dat_file(
    path: str,
    expected_length: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Robustly parse plain-text .dat files (one float per line).
    - Skips blank lines and lines starting with '#'
    - Tolerates CRLF (\\r\\n) and LF (\\n) line endings
    - Tracks and counts parse errors
    - Raises ValueError if no valid floats could be parsed
    - Zero-pads or truncates to expected_length with a logged warning if specified
    - Returns a 1D float32 numpy array
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    values = []
    parse_errors = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            try:
                val = float(cleaned)
                values.append(val)
            except ValueError:
                parse_errors += 1
                if verbose:
                    logger.warning("Parse error at line %d in %s: '%s'", line_num, path, cleaned)

    if parse_errors > 0:
        logger.warning("Encountered %d unparseable lines in %s", parse_errors, path)

    if len(values) == 0:
        raise ValueError(f"No valid numeric data found in file: {path}")

    arr = np.asarray(values, dtype=np.float32)

    if expected_length is not None:
        curr_len = len(arr)
        if curr_len < expected_length:
            logger.warning(
                "File %s has length %d < expected %d. Zero-padding.",
                path, curr_len, expected_length
            )
            pad_width = expected_length - curr_len
            arr = np.pad(arr, (0, pad_width), mode="constant", constant_values=0.0)
        elif curr_len > expected_length:
            logger.warning(
                "File %s has length %d > expected %d. Truncating.",
                path, curr_len, expected_length
            )
            arr = arr[:expected_length]

    return arr


def save_dat_file(path: str, signal: Union[np.ndarray, torch.Tensor]) -> None:
    """
    Save 1D signal to a plain-text .dat file (one float per line).
    """
    ensure_dir(path)
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().squeeze().numpy()
    signal = np.asarray(signal, dtype=np.float32).ravel()

    with open(path, "w", encoding="utf-8") as f:
        for val in signal:
            f.write(f"{val:.8e}\n")


def compute_rms_envelope(signal: np.ndarray, window: int) -> np.ndarray:
    """
    Compute moving RMS envelope in O(N) time using cumulative sums with edge padding.
    Output has identical length to input signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return np.array([], dtype=np.float32)
    if window <= 1:
        return np.abs(signal).astype(np.float32)

    window = min(window, n)
    sq = signal ** 2

    # Symmetrical edge padding
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    sq_padded = np.pad(sq, (pad_left, pad_right), mode="edge")

    # Cumulative sum for O(N) moving window calculation
    cumsum = np.cumsum(np.concatenate(([0.0], sq_padded)))
    window_sum = cumsum[window: window + n] - cumsum[:n]
    mean_sq = np.maximum(window_sum / float(window), 0.0)
    rms = np.sqrt(mean_sq)
    return rms.astype(np.float32)


def estimate_noise_floor(rms: np.ndarray, percentile: float = 20.0) -> float:
    """
    Estimate background noise floor as a low percentile of the RMS envelope.
    """
    if len(rms) == 0:
        return 0.0
    return float(np.percentile(rms, percentile))


def detect_active_region(
    rms: np.ndarray,
    noise_floor: float,
    threshold_factor: float,
    pad_samples: int,
    signal_length: int,
) -> Tuple[int, int]:
    """
    Identify (start, end) index bounds of active pulse region.
    Pads detected region and clamps within [0, signal_length].
    Falls back to (0, signal_length) with a warning if no samples exceed threshold.
    """
    threshold = noise_floor * threshold_factor
    active_indices = np.where(rms > threshold)[0]

    if len(active_indices) == 0:
        logger.warning(
            "No samples exceeded threshold (factor=%.2f, floor=%.4e). "
            "Falling back to entire signal length.",
            threshold_factor, noise_floor
        )
        return 0, signal_length

    raw_start = int(np.min(active_indices))
    raw_end = int(np.max(active_indices)) + 1

    start = max(0, raw_start - pad_samples)
    end = min(signal_length, raw_end + pad_samples)

    return start, end


def extract_active_region(
    signal: np.ndarray,
    cfg: Config,
    target_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract active region and noise segment from a raw recording.
    Automatically selects trial vs synthetic parameters based on signal length
    or configuration if not explicitly provided.
    Returns:
        dict with keys:
            'active': np.ndarray padded/truncated to target_length
            'noise': np.ndarray concatenated segments outside active window
            'start': int start index
            'end': int end index
            'noise_floor': float
            'threshold': float
    """
    signal = np.asarray(signal, dtype=np.float32)
    sig_len = len(signal)

    # Determine parameter set (trial data is ~19.3k samples or sampled at 17.8 kHz)
    is_trial = sig_len < 22000
    rms_window = cfg.TRIAL_RMS_WINDOW if is_trial else cfg.EXTRACTION_RMS_WINDOW
    pad_samples = cfg.TRIAL_PAD_SAMPLES if is_trial else cfg.EXTRACTION_PAD_SAMPLES
    out_length = target_length or (cfg.TRIAL_SIGNAL_LENGTH if is_trial else cfg.SIGNAL_LENGTH)
    threshold_factor = cfg.EXTRACTION_THRESHOLD_FACTOR
    noise_percentile = cfg.EXTRACTION_NOISE_PERCENTILE

    rms = compute_rms_envelope(signal, rms_window)
    noise_floor = estimate_noise_floor(rms, noise_percentile)
    threshold = noise_floor * threshold_factor
    start, end = detect_active_region(rms, noise_floor, threshold_factor, pad_samples, sig_len)

    active_raw = signal[start:end]
    act_len = len(active_raw)

    if act_len < out_length:
        active = np.pad(active_raw, (0, out_length - act_len), mode="constant", constant_values=0.0)
    elif act_len > out_length:
        active = active_raw[:out_length]
    else:
        active = active_raw

    # Concatenate inactive segments outside [start, end]
    noise_parts = []
    if start > 0:
        noise_parts.append(signal[:start])
    if end < sig_len:
        noise_parts.append(signal[end:])

    if noise_parts:
        noise_seg = np.concatenate(noise_parts)
    else:
        noise_seg = signal.copy()

    return {
        "active": active.astype(np.float32),
        "noise": noise_seg.astype(np.float32),
        "start": start,
        "end": end,
        "noise_floor": float(noise_floor),
        "threshold": float(threshold),
    }


def extract_noise_reference(signal: np.ndarray, window_len: int) -> np.ndarray:
    """
    Extract lowest-energy sliding window of length window_len in O(N) time.
    Used as robust fallback when inactive regions are constrained.
    """
    signal = np.asarray(signal, dtype=np.float32)
    n = len(signal)
    if n <= window_len:
        return np.pad(signal, (0, max(0, window_len - n)), mode="constant")

    sq = (signal.astype(np.float64)) ** 2
    cumsum = np.cumsum(np.concatenate(([0.0], sq)))
    win_energy = cumsum[window_len:] - cumsum[:-window_len]
    min_idx = int(np.argmin(win_energy))
    return signal[min_idx: min_idx + window_len].astype(np.float32)


def normalize_signal(signal: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float, float]:
    """
    Z-score normalize a 1-D signal: (x - mean) / (std + eps).
    Returns (normalized_signal, mean, std).
    """
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    norm = (signal - mean) / (std + eps)
    return norm.astype(np.float32), mean, std


def denormalize_signal(normalized: Union[np.ndarray, torch.Tensor], mean: float, std: float) -> np.ndarray:
    """
    Reverse z-score normalization: x * std + mean.
    """
    if isinstance(normalized, torch.Tensor):
        normalized = normalized.detach().cpu().squeeze().numpy()
    return (normalized * std + mean).astype(np.float32)


def to_tensor(signal: Union[np.ndarray, list], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert 1-D numpy array or list to PyTorch 2-D tensor of shape (1, N).
    """
    arr = np.asarray(signal, dtype=np.float32)
    t = torch.from_numpy(arr).to(dtype=dtype)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

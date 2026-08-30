"""
utils.py
--------
Shared utility functions for the NSTL Acoustic Pipeline.

Provides:
  - set_seed(seed)                  reproducible runs
  - load_dat_file(path, ...)        robust .dat loader (text, one float per line)
  - extract_noise_reference(signal) lowest-energy sub-window noise estimator
  - normalize_signal(signal)        z-score normalisation
  - to_tensor(array)                numpy → 1-channel PyTorch tensor
  - ensure_dir(path)                mkdir -p equivalent

All functions are stateless and import-safe.
"""

import os
import random
import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducible runs.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic : bool
        If True, enable deterministic CUDA algorithms (slower but fully
        reproducible on GPU).  Ignored when CUDA is unavailable.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not available; numpy/random still seeded


# --------------------------------------------------------------------------- #
# .dat file loading
# --------------------------------------------------------------------------- #

def load_dat_file(
    path: str,
    expected_length: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Load a .dat file containing one float value per line (plain-text format).

    The loader is intentionally robust:
      - Skips blank lines.
      - Skips comment lines (starting with '#').
      - Handles both Unix and Windows line endings.
      - If the resulting array is shorter than ``expected_length``, it is
        zero-padded with a warning.
      - If it is longer than ``expected_length``, it is truncated with a
        warning.

    Parameters
    ----------
    path : str
        Absolute or relative path to the .dat file.
    expected_length : int, optional
        If given, the returned array is guaranteed to have this exact length.
    verbose : bool
        If True, print file-loading diagnostics.

    Returns
    -------
    np.ndarray
        1-D float32 array of signal samples.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file exists but contains no parseable numeric values.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[load_dat_file] File not found: {path!r}")

    samples = []
    parse_errors = 0

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                samples.append(float(line))
            except ValueError:
                parse_errors += 1
                if verbose:
                    print(
                        f"  [load_dat_file] Parse error at line {line_no} in"
                        f" {os.path.basename(path)!r}: {line!r}"
                    )

    if not samples:
        raise ValueError(
            f"[load_dat_file] No parseable values found in {path!r}. "
            "Check the file format."
        )

    arr = np.asarray(samples, dtype=np.float32)
    raw_len = len(arr)

    if verbose:
        print(
            f"  [load_dat_file] {os.path.basename(path)!r}: "
            f"loaded {raw_len} samples"
            + (f", {parse_errors} parse errors ignored" if parse_errors else "")
        )

    if expected_length is not None and raw_len != expected_length:
        if raw_len < expected_length:
            pad = expected_length - raw_len
            arr = np.pad(arr, (0, pad), mode="constant")
            log.warning(
                "Signal in %r is shorter than expected (%d < %d); "
                "zero-padded by %d samples.",
                path, raw_len, expected_length, pad,
            )
        else:
            arr = arr[:expected_length]
            log.warning(
                "Signal in %r is longer than expected (%d > %d); "
                "truncated to %d samples.",
                path, raw_len, expected_length, expected_length,
            )

    return arr


# --------------------------------------------------------------------------- #
# Noise reference extraction
# --------------------------------------------------------------------------- #

def extract_noise_reference(
    signal: np.ndarray,
    window_len: int = 512,
) -> np.ndarray:
    """
    Extract a noise-reference segment from a signal using the lowest-energy
    sliding window.

    This is used when there is NO guaranteed noise-only prefix in the trial
    data.  The window with the minimum RMS energy is returned as the noise
    reference.

    Parameters
    ----------
    signal : np.ndarray
        1-D float array, shape (N,).
    window_len : int
        Length of the sliding window in samples.

    Returns
    -------
    np.ndarray
        1-D float32 array, shape (window_len,), representing the noise
        reference segment.

    Raises
    ------
    ValueError
        If the signal is shorter than ``window_len``.
    """
    n = len(signal)
    if n < window_len:
        raise ValueError(
            f"[extract_noise_reference] Signal length ({n}) is shorter than "
            f"the requested window ({window_len}).  Reduce NOISE_REF_WINDOW in Config."
        )

    # Compute squared energy for each window using cumulative-sum trick (O(N)).
    sig_sq = signal.astype(np.float64) ** 2
    cum = np.cumsum(sig_sq)
    # Energy of window starting at index i: cum[i+W-1] - (cum[i-1] if i>0 else 0)
    starts = np.arange(n - window_len + 1)
    energies = cum[starts + window_len - 1] - np.concatenate(([[0.0], cum[:-1]]))[starts]

    best_start = int(np.argmin(energies))
    return signal[best_start : best_start + window_len].astype(np.float32)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def normalize_signal(
    signal: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, float, float]:
    """
    Z-score normalise a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D float array.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    normalised : np.ndarray
        Zero-mean, unit-variance signal.
    mean : float
        Original mean (for de-normalisation).
    std : float
        Original std (for de-normalisation).
    """
    mean = float(np.mean(signal))
    std = float(np.std(signal)) + eps
    return (signal - mean) / std, mean, std


def denormalize_signal(
    signal: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Reverse z-score normalisation."""
    return signal * std + mean


# --------------------------------------------------------------------------- #
# Tensor helpers
# --------------------------------------------------------------------------- #

def to_tensor(array: np.ndarray):
    """
    Convert a 1-D numpy array to a (1, N) float32 PyTorch tensor suitable
    for a 1-channel 1-D convolutional model.

    Requires PyTorch.
    """
    import torch
    arr = np.asarray(array, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0)   # shape: (1, N)


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #

def ensure_dir(path: str) -> None:
    """Create ``path`` (and any missing parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)

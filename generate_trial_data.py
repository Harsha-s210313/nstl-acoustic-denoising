"""
Synthetic and Trial Data Generator for Underwater Acoustic Pipeline.
Generates reproducible LFM (chirp) pulses and ambient acoustic noise
stored as plain-text .dat files (one float per line).
"""

import os
import argparse
import logging
from typing import Optional
import numpy as np

try:
    from .config import Config, default_config
    from .utils import ensure_dir, set_seed
except (ImportError, ValueError):
    from config import Config, default_config
    from utils import ensure_dir, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_lfm_pulse(
    fs: float,
    duration_samples: int,
    pulse_samples: int,
    start_sample: int,
    f_start: float,
    f_end: float,
    amplitude: float = 1.0,
    decay_rate: float = 0.0,
) -> np.ndarray:
    """
    Generate a 1-D Linear Frequency Modulated (LFM) chirp pulse inside a recording.
    """
    total = np.zeros(duration_samples, dtype=np.float32)
    end_sample = min(duration_samples, start_sample + pulse_samples)
    actual_pulse_len = end_sample - start_sample
    if actual_pulse_len <= 0:
        return total

    t = np.arange(actual_pulse_len, dtype=np.float32) / fs
    t_pulse = pulse_samples / fs

    # Chirp rate k = (f_end - f_start) / t_pulse
    k = (f_end - f_start) / max(t_pulse, 1e-6)
    # Instantaneous phase = 2 * pi * (f_start * t + 0.5 * k * t^2)
    phase = 2.0 * np.pi * (f_start * t + 0.5 * k * (t ** 2))
    pulse = amplitude * np.sin(phase)

    # Smooth Tukey-like window at pulse edges to avoid artificial hard transients
    window = np.hanning(actual_pulse_len)
    pulse = pulse * window

    if decay_rate > 0.0:
        decay = np.exp(-decay_rate * t)
        pulse = pulse * decay

    total[start_sample:end_sample] = pulse
    return total


def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.RandomState,
    colored: bool = False,
) -> np.ndarray:
    """
    Add Gaussian or colored ambient noise to signal matching target SNR in dB.
    """
    sig_power = float(np.mean(signal ** 2))
    if sig_power < 1e-12:
        sig_power = 1.0

    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, 1.0, size=len(signal)).astype(np.float32)

    if colored:
        # Simple AR(1) low-frequency acoustic noise coloration
        alpha = 0.8
        for i in range(1, len(noise)):
            noise[i] = alpha * noise[i - 1] + (1 - alpha) * noise[i]

    cur_noise_power = float(np.mean(noise ** 2)) + 1e-12
    noise = noise * np.sqrt(target_noise_power / cur_noise_power)
    return signal + noise


def write_dat_with_optional_artefacts(
    path: str,
    data: np.ndarray,
    inject_artefacts: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> None:
    """
    Write 1D numpy array as a text .dat file.
    If inject_artefacts is True, injects comments, CRLF, and rare parse test lines.
    """
    ensure_dir(path)
    r = rng or np.random.RandomState(42)

    with open(path, "w", encoding="utf-8", newline="\r\n" if inject_artefacts else "\n") as f:
        if inject_artefacts:
            f.write("# NSTL Hydrophone Trial Acquisition Log\n")
            f.write("# Calibration factor: 1.0\n")
            f.write("\n")  # Blank line

        for idx, val in enumerate(data):
            if inject_artefacts and idx == 50 and r.random() < 0.2:
                f.write("# Intermediate channel status check: OK\n")
            if inject_artefacts and idx == 100 and r.random() < 0.1:
                f.write("PARSE_ERROR_INVALID_SAMPLE\n")  # Test robustness against non-numeric lines
            f.write(f"{val:.8e}\n")


def generate_all_data(
    cfg: Optional[Config] = None,
    num_synthetic: int = 24,
    num_detector_noise: int = 24,
    num_trial_pri: int = 3,
    channels_per_pri: int = 4,
    artefacts: bool = False,
) -> None:
    """
    Generate synthetic dataset, detector dataset, and realistic trial dataset.
    """
    config = cfg or default_config
    config.validate()
    rng = np.random.RandomState(config.RANDOM_SEED)

    logger.info("Generating Synthetic Clean and Noisy Pairs (%d pairs)...", num_synthetic)
    clean_dir = os.path.join(config.SYNTHETIC_DATA_DIR, config.SYNTHETIC_CLEAN_SUBDIR)
    noisy_dir = os.path.join(config.SYNTHETIC_DATA_DIR, config.SYNTHETIC_NOISY_SUBDIR)
    det_sig_dir = os.path.join(config.DETECTOR_DATA_DIR, "signal")
    det_noise_dir = os.path.join(config.DETECTOR_DATA_DIR, "noise")

    ensure_dir(clean_dir)
    ensure_dir(noisy_dir)
    ensure_dir(det_sig_dir)
    ensure_dir(det_noise_dir)

    # 1. Synthetic 100 kHz datasets
    fs = config.SAMPLE_RATE
    total_len = config.SIGNAL_LENGTH

    for i in range(1, num_synthetic + 1):
        filename = f"synth_{i:04d}.dat"
        pulse_len = int(rng.uniform(4000, 8000))
        start_idx = int(rng.uniform(1000, total_len - pulse_len - 1000))
        f_start = float(rng.uniform(15000, 22000))
        f_end = float(rng.uniform(28000, 38000))
        amp = float(rng.uniform(0.7, 1.5))
        snr_db = float(rng.uniform(-3.0, 12.0))

        clean_sig = generate_lfm_pulse(
            fs=fs,
            duration_samples=total_len,
            pulse_samples=pulse_len,
            start_sample=start_idx,
            f_start=f_start,
            f_end=f_end,
            amplitude=amp,
        )
        noisy_sig = add_noise(clean_sig, snr_db=snr_db, rng=rng, colored=(i % 2 == 0))

        if artefacts and i % 5 == 0:
            # Simulate slight sensor clipping artefact
            clip_val = float(0.85 * np.max(np.abs(noisy_sig)))
            noisy_sig = np.clip(noisy_sig, -clip_val, clip_val)

        write_dat_with_optional_artefacts(os.path.join(clean_dir, filename), clean_sig, inject_artefacts=False)
        write_dat_with_optional_artefacts(os.path.join(noisy_dir, filename), noisy_sig, inject_artefacts=artefacts, rng=rng)

        # Populate detector signal dataset
        write_dat_with_optional_artefacts(os.path.join(det_sig_dir, filename), noisy_sig, inject_artefacts=artefacts, rng=rng)

    # 2. Pure ambient noise files for detector
    for i in range(1, num_detector_noise + 1):
        filename = f"det_noise_{i:04d}.dat"
        noise = rng.normal(0.0, float(rng.uniform(0.2, 0.8)), size=total_len).astype(np.float32)
        write_dat_with_optional_artefacts(os.path.join(det_noise_dir, filename), noise, inject_artefacts=artefacts, rng=rng)

    logger.info("Generating Trial Datasets (17.8 kHz bandpass-sampled recordings)...")
    # 3. Realistic trial recordings: ~19,299 samples, ~113 ms pulse (~2011 samples)
    trial_fs = config.TRIAL_SAMPLE_RATE
    trial_rec_len = 19299
    trial_pulse_len = 2011

    trial_sig_dir = (
        config.TRIAL_SIGNAL_SUBDIR
        if os.path.isabs(config.TRIAL_SIGNAL_SUBDIR)
        else os.path.join(config.TRIAL_DATA_DIR, config.TRIAL_SIGNAL_SUBDIR)
    )

    for p in range(1, num_trial_pri + 1):
        pri_name = f"PRI_{p:02d}"
        for ch in range(1, channels_per_pri + 1):
            ch_name = f"channel{ch:03d}.dat"
            ch_path = os.path.join(trial_sig_dir, pri_name, ch_name)

            # Active pulse around middle (e.g. index 6000 to 12000)
            start_idx = int(rng.uniform(5000, 11000))
            f_start = float(rng.uniform(2500, 4000))
            f_end = float(rng.uniform(5500, 7500))
            amp = float(rng.uniform(0.8, 1.8))
            snr_db = float(rng.uniform(-2.0, 10.0))

            pulse = generate_lfm_pulse(
                fs=trial_fs,
                duration_samples=trial_rec_len,
                pulse_samples=trial_pulse_len,
                start_sample=start_idx,
                f_start=f_start,
                f_end=f_end,
                amplitude=amp,
            )
            trial_rec = add_noise(pulse, snr_db=snr_db, rng=rng, colored=True)

            if artefacts and (p == 1 and ch == 1):
                # Truncated recording to verify parser and extraction robustness
                trial_rec = trial_rec[:17500]

            write_dat_with_optional_artefacts(ch_path, trial_rec, inject_artefacts=artefacts, rng=rng)

    logger.info("Data generation complete. All datasets ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic & Trial Hydrophone Datasets")
    parser.add_argument("--num-synthetic", type=int, default=24, help="Number of synthetic clean/noisy pairs")
    parser.add_argument("--num-detector-noise", type=int, default=24, help="Number of pure noise files for detector")
    parser.add_argument("--num-trial-pri", type=int, default=3, help="Number of trial PRIs")
    parser.add_argument("--channels-per-pri", type=int, default=4, help="Channels per trial PRI")
    parser.add_argument("--artefacts", action="store_true", help="Inject realistic artefacts (clipping, comments, CRLF)")
    args = parser.parse_args()

    generate_all_data(
        num_synthetic=args.num_synthetic,
        num_detector_noise=args.num_detector_noise,
        num_trial_pri=args.num_trial_pri,
        channels_per_pri=args.channels_per_pri,
        artefacts=args.artefacts,
    )

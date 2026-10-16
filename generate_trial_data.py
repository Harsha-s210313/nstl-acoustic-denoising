"""
Synthetic and Trial Data Generator for Underwater Acoustic Pipeline.
Faithfully replicates real sea trial hydrophone acquisitions matching:
- High-amplitude dynamic range (+/-450 to +/-650 peak amplitude)
- Bandpass-sampled LFM chirp pulse (~113 ms, 2011 samples at 17.8 kHz)
- Dense acoustic multipath reverberation tail (surface & seabed echo clusters)
- Colored ambient ocean background noise (std ~14-18 RMS)
- Impulsive acoustic clicks / hydrophone sensor transients (+/-150 to +/-380)
- Exact plain-text formatting with 6 leading zeros matching real trial data.
"""

import os
import argparse
import logging
from typing import Optional, Tuple
import numpy as np
from scipy.signal import butter, lfilter

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

    k = (f_end - f_start) / max(t_pulse, 1e-6)
    phase = 2.0 * np.pi * (f_start * t + 0.5 * k * (t ** 2))
    pulse = amplitude * np.sin(phase)

    # Smooth Hanning window at pulse edges to match acoustic transducer response
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
            f.write("\n")

        for idx, val in enumerate(data):
            if inject_artefacts and idx == 50 and r.random() < 0.2:
                f.write("# Intermediate channel status check: OK\n")
            if inject_artefacts and idx == 100 and r.random() < 0.1:
                f.write("PARSE_ERROR_INVALID_SAMPLE\n")
            f.write(f"{val:.6f}\n")


def generate_physics_based_trial_recording(
    fs: float = 17800.0,
    total_len: int = 19299,
    pulse_len: int = 2011,
    start_sample: Optional[int] = None,
    f_start: float = 3200.0,
    f_end: float = 5700.0,
    amplitude: Optional[float] = None,
    noise_std: float = 14.0,
    rng: Optional[np.random.RandomState] = None,
    leading_zeros: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize an underwater acoustic recording physically replicating real sea trial data:
    - Bandpass sampling rate: 17,800 Hz
    - Length: 19,299 samples (~1.084 seconds)
    - Direct LFM chirp pulse: ~113 ms (2011 samples) with smooth Hanning tapering
    - Peak amplitude: +/-420 to +/-580
    - Discrete multipath reverberation cluster: surface and seabed echo arrivals
      decaying exponentially over 400 ms (samples 10,500 to 18,500)
    - Realistic colored ocean background noise (std ~14-18 RMS)
    - Sparse impulsive acoustic clicks / sensor transients (+/-150 to +/-380)
    - Exact leading zeros (6 zeros) matching real trial recordings.

    Returns:
        (clean_direct_signal, noisy_reverberant_signal)
    """
    r = rng or np.random.RandomState(42)

    amp = amplitude if amplitude is not None else float(r.uniform(420.0, 580.0))
    if start_sample is None:
        start_idx = int(r.uniform(8500, 9500))
    else:
        start_idx = max(leading_zeros + 10, min(start_sample, total_len - pulse_len - 100))

    # 1. Direct LFM chirp pulse
    t = np.arange(pulse_len, dtype=np.float32) / fs
    k = (f_end - f_start) / (pulse_len / fs)
    phase = 2.0 * np.pi * (f_start * t + 0.5 * k * (t ** 2))
    direct_pulse = (amp * np.sin(phase) * np.hanning(pulse_len)).astype(np.float32)

    clean_sig = np.zeros(total_len, dtype=np.float32)
    end_idx = min(total_len, start_idx + pulse_len)
    clean_sig[start_idx:end_idx] = direct_pulse[: end_idx - start_idx]

    # 2. Discrete multipath reverberation tail (surface / bottom bounces)
    reverb = np.zeros(total_len, dtype=np.float32)
    multipath_delays = [220, 550, 1100, 1750, 2500, 3400, 4500, 5800, 7200]
    for d in multipath_delays:
        arr_idx = start_idx + d
        if arr_idx + pulse_len <= total_len:
            decay = float(np.exp(-d / 3200.0))
            refl_amp = amp * decay * float(r.uniform(0.25, 0.55)) * float(r.choice([-1.0, 1.0]))
            refl_phase = phase + float(r.uniform(0.0, 2.0 * np.pi))
            reverb[arr_idx : arr_idx + pulse_len] += (
                refl_amp * np.sin(refl_phase) * np.hanning(pulse_len)
            ).astype(np.float32)

    # 3. Colored ocean background noise (std ~14-18 RMS)
    nyq = 0.5 * fs
    low_cutoff = max(0.01, 1500.0 / nyq)
    high_cutoff = min(0.99, 7500.0 / nyq)
    b, a = butter(2, [low_cutoff, high_cutoff], btype="bandpass")
    white = r.normal(0.0, noise_std, size=total_len).astype(np.float32)
    ocean_noise = (lfilter(b, a, white) * 1.4).astype(np.float32)

    # 4. Sparse impulsive acoustic clicks / sensor spikes (+/-150 to +/-380)
    spikes = np.zeros(total_len, dtype=np.float32)
    num_spikes = r.randint(6, 12)
    spike_indices = r.choice(np.arange(100, total_len - 100), size=num_spikes, replace=False)
    for s_idx in spike_indices:
        spikes[s_idx] = float(r.uniform(150.0, 380.0) * r.choice([-1.0, 1.0]))

    noisy_sig = clean_sig + reverb + ocean_noise + spikes

    # 5. Exact leading zeros suppression
    if leading_zeros > 0:
        clean_sig[:leading_zeros] = 0.0
        noisy_sig[:leading_zeros] = 0.0

    return clean_sig, noisy_sig


def generate_physics_based_synthetic_pair(
    fs: float = 100000.0,
    total_len: int = 25000,
    pulse_samples: Optional[int] = None,
    start_sample: Optional[int] = None,
    f_start: Optional[float] = None,
    f_end: Optional[float] = None,
    amplitude: Optional[float] = None,
    snr_db: Optional[float] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a clean and noisy pair for Stage 2A U-Net pretraining.
    Ground truth clean: Direct LFM chirp pulse (zero noise, zero reverberation).
    Noisy: Direct pulse + multipath reverberation echoes + colored ocean noise + acoustic clicks.
    Trains the U-Net for joint denoising and dereverberation.
    """
    r = rng or np.random.RandomState(42)

    p_len = pulse_samples if pulse_samples is not None else int(r.uniform(3000, 6000))
    if start_sample is None:
        start_idx = int(r.uniform(2000, max(2001, total_len - p_len - 6000)))
    else:
        start_idx = max(100, min(start_sample, total_len - p_len - 100))

    f1 = f_start if f_start is not None else float(r.uniform(18000.0, 22000.0))
    f2 = f_end if f_end is not None else float(r.uniform(25000.0, 32000.0))
    amp = amplitude if amplitude is not None else float(r.uniform(350.0, 550.0))

    # 1. Direct chirp pulse
    t = np.arange(p_len, dtype=np.float32) / fs
    k = (f2 - f1) / (p_len / fs)
    phase = 2.0 * np.pi * (f1 * t + 0.5 * k * (t ** 2))
    direct = (amp * np.sin(phase) * np.hanning(p_len)).astype(np.float32)

    clean_sig = np.zeros(total_len, dtype=np.float32)
    end_idx = min(total_len, start_idx + p_len)
    clean_sig[start_idx:end_idx] = direct[: end_idx - start_idx]

    # 2. Multipath reverberation echoes
    reverb = np.zeros(total_len, dtype=np.float32)
    delays = [300, 750, 1500, 2500, 3800, 5200]
    for d in delays:
        arr_idx = start_idx + d
        if arr_idx + p_len <= total_len:
            decay = float(np.exp(-d / 3000.0))
            refl_amp = amp * decay * float(r.uniform(0.20, 0.50)) * float(r.choice([-1.0, 1.0]))
            refl_phase = phase + float(r.uniform(0.0, 2.0 * np.pi))
            reverb[arr_idx : arr_idx + p_len] += (
                refl_amp * np.sin(refl_phase) * np.hanning(p_len)
            ).astype(np.float32)

    # 3. Colored ocean noise
    nyq = 0.5 * fs
    low_cutoff = max(0.01, 10000.0 / nyq)
    high_cutoff = min(0.99, 40000.0 / nyq)
    b, a = butter(2, [low_cutoff, high_cutoff], btype="bandpass")
    target_snr = snr_db if snr_db is not None else float(r.uniform(-2.0, 10.0))
    sig_power = float(np.mean(direct ** 2))
    noise_power = sig_power / (10.0 ** (target_snr / 10.0))
    white = r.normal(0.0, np.sqrt(max(1.0, noise_power)), size=total_len).astype(np.float32)
    ocean_noise = lfilter(b, a, white).astype(np.float32)

    # 4. Sparse acoustic spikes
    spikes = np.zeros(total_len, dtype=np.float32)
    num_spikes = r.randint(4, 10)
    spike_indices = r.choice(np.arange(100, total_len - 100), size=num_spikes, replace=False)
    for s_idx in spike_indices:
        spikes[s_idx] = float(r.uniform(100.0, 300.0) * r.choice([-1.0, 1.0]))

    noisy_sig = clean_sig + reverb + ocean_noise + spikes
    return clean_sig, noisy_sig


def generate_all_data(
    cfg: Optional[Config] = None,
    num_synthetic: int = 32,
    num_detector_noise: int = 32,
    num_trial_pri: int = 3,
    channels_per_pri: int = 4,
    artefacts: bool = False,
    plot_sample: bool = False,
) -> None:
    """
    Generate synthetic dataset, detector dataset, and realistic trial dataset.
    All data is generated with physically realistic ocean acoustic properties.
    """
    config = cfg or default_config
    config.validate()
    rng = np.random.RandomState(config.RANDOM_SEED)

    logger.info("Generating Physics-Based Synthetic Clean and Noisy Pairs (%d pairs)...", num_synthetic)
    clean_dir = os.path.join(config.SYNTHETIC_DATA_DIR, config.SYNTHETIC_CLEAN_SUBDIR)
    noisy_dir = os.path.join(config.SYNTHETIC_DATA_DIR, config.SYNTHETIC_NOISY_SUBDIR)
    det_sig_dir = os.path.join(config.DETECTOR_DATA_DIR, "signal")
    det_noise_dir = os.path.join(config.DETECTOR_DATA_DIR, "noise")

    ensure_dir(clean_dir)
    ensure_dir(noisy_dir)
    ensure_dir(det_sig_dir)
    ensure_dir(det_noise_dir)

    # 1. Synthetic clean/noisy pairs for pretraining
    fs = config.SAMPLE_RATE
    total_len = config.SIGNAL_LENGTH

    for i in range(1, num_synthetic + 1):
        filename = f"synth_{i:04d}.dat"
        clean_sig, noisy_sig = generate_physics_based_synthetic_pair(
            fs=fs,
            total_len=total_len,
            rng=rng,
        )

        if artefacts and i % 5 == 0:
            clip_val = float(0.85 * np.max(np.abs(noisy_sig)))
            noisy_sig = np.clip(noisy_sig, -clip_val, clip_val)

        write_dat_with_optional_artefacts(os.path.join(clean_dir, filename), clean_sig, inject_artefacts=False)
        write_dat_with_optional_artefacts(os.path.join(noisy_dir, filename), noisy_sig, inject_artefacts=artefacts, rng=rng)

        # Populate detector signal dataset
        write_dat_with_optional_artefacts(os.path.join(det_sig_dir, filename), noisy_sig, inject_artefacts=artefacts, rng=rng)

    # 2. Pure ambient ocean noise files for detector (negative class)
    logger.info("Generating Pure Ocean Noise Recordings (%d files)...", num_detector_noise)
    nyq = 0.5 * fs
    b, a = butter(2, [10000.0 / nyq, 40000.0 / nyq], btype="bandpass")
    for i in range(1, num_detector_noise + 1):
        filename = f"det_noise_{i:04d}.dat"
        white = rng.normal(0.0, float(rng.uniform(15.0, 35.0)), size=total_len).astype(np.float32)
        ocean_noise = lfilter(b, a, white).astype(np.float32)

        # Add occasional biological click
        if rng.random() < 0.3:
            s_idx = rng.randint(100, total_len - 100)
            ocean_noise[s_idx] = float(rng.uniform(100.0, 250.0) * rng.choice([-1.0, 1.0]))

        write_dat_with_optional_artefacts(os.path.join(det_noise_dir, filename), ocean_noise, inject_artefacts=artefacts, rng=rng)

    # 3. Realistic trial recordings matching Image 1 & sea trial dataset
    logger.info("Generating Trial Datasets (17.8 kHz bandpass-sampled recordings, 19,299 samples)...")
    trial_fs = config.TRIAL_SAMPLE_RATE
    trial_rec_len = 19299
    trial_pulse_len = 2011

    # Inter-channel arrival offsets matching physical hydrophone array
    channel_offsets = [5300, 0, 170, 240, 640, 775, 5330, 0, 172, 173, 640]

    trial_sig_dir = os.path.join(config.SYNTHETIC_DATA_DIR, "trial_data", "signal")
    ensure_dir(trial_sig_dir)

    sample_to_plot = None

    for p in range(1, num_trial_pri + 1):
        pri_name = f"PRI_{p:02d}"
        for ch in range(1, channels_per_pri + 1):
            ch_name = f"channel{ch:03d}.dat"
            ch_path = os.path.join(trial_sig_dir, pri_name, ch_name)

            offset = channel_offsets[(ch - 1) % len(channel_offsets)] + rng.randint(-15, 15)
            base_arrival = 9150 + offset

            clean_rec, trial_rec = generate_physics_based_trial_recording(
                fs=trial_fs,
                total_len=trial_rec_len,
                pulse_len=trial_pulse_len,
                start_sample=base_arrival,
                amplitude=float(rng.uniform(420.0, 580.0)),
                noise_std=float(rng.uniform(12.0, 17.0)),
                rng=rng,
                leading_zeros=6,
            )

            if artefacts and (p == 1 and ch == 1):
                trial_rec = trial_rec[:17500]

            write_dat_with_optional_artefacts(ch_path, trial_rec, inject_artefacts=artefacts, rng=rng)

            if sample_to_plot is None:
                sample_to_plot = trial_rec.copy()

    # 4. Optional visual plot verification
    if plot_sample and sample_to_plot is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(sample_to_plot, color="#1f77b4", linewidth=0.7)
            ax.set_ylim(-700, 700)
            ax.set_xlim(0, 19299)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_title("Physically Replicated Sea Trial Data (17.8 kHz, 19,299 samples)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            plt.tight_layout()
            plot_file = os.path.join(config.SYNTHETIC_DATA_DIR, "replicated_trial_sample.png")
            plt.savefig(plot_file, dpi=150)
            plt.close(fig)
            logger.info("Saved verification plot to: %s", plot_file)
        except Exception as e:
            logger.warning("Could not generate verification plot: %s", e)

    logger.info("Data generation complete. All datasets ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic & Trial Hydrophone Datasets")
    parser.add_argument("--num-synthetic", type=int, default=32, help="Number of synthetic clean/noisy pairs")
    parser.add_argument("--num-detector-noise", type=int, default=32, help="Number of pure noise files for detector")
    parser.add_argument("--num-trial-pri", type=int, default=3, help="Number of trial PRIs")
    parser.add_argument("--channels-per-pri", type=int, default=4, help="Channels per trial PRI")
    parser.add_argument("--artefacts", action="store_true", help="Inject realistic artefacts (clipping, comments, CRLF)")
    parser.add_argument("--plot-sample", action="store_true", help="Save visual verification plot of replicated sample")
    args = parser.parse_args()

    generate_all_data(
        num_synthetic=args.num_synthetic,
        num_detector_noise=args.num_detector_noise,
        num_trial_pri=args.num_trial_pri,
        channels_per_pri=args.channels_per_pri,
        artefacts=args.artefacts,
        plot_sample=args.plot_sample,
    )

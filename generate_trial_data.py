"""
generate_trial_data.py
-----------------------
Generates 10 synthetic trial-like .dat files that mimic real hydrophone recordings.

Each file:
  - 400,000 samples @ 100 kHz  =  4 seconds total
  - AWGN noise throughout  (noise floor)
  - A randomly-placed LFM (Linear Frequency Modulated) pulse
    representing the active signal+noise region
  - A weak reverb echo ~50 ms after the main pulse tail
  - Random SNR variation across files (so the extractor is tested on real diversity)

After generating, each file is passed through signal_extractor and a verification
plot is shown + saved. Close each plot window to move to the next file.

Run from IDLE: open this file and press F5.
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime

# ================================================================== #
#                  USER CONFIGURATION                                 #
# ================================================================== #

# Where to save the generated .dat files
OUTPUT_DIR = r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\data\trial_data\Signal\PRI_TEST"

# Where to save verification plots
PLOT_DIR = r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\inference_output\verification"

N_FILES        = 10            # number of files to generate
FS             = 100_000       # sampling rate (Hz)
TOTAL_SAMPLES  = 400_000       # samples per file (4 seconds)

# LFM pulse parameters
LFM_DURATION_MS  = 100.0       # pulse width  (ms)
LFM_F_START      = 500.0       # chirp start frequency (Hz)
LFM_F_END        = 4000.0      # chirp end   frequency (Hz)

# Amplitude
NOISE_STD        = 0.10        # AWGN noise standard deviation throughout
SIGNAL_AMP_MIN   = 3.0         # minimum signal amplitude (× noise std)
SIGNAL_AMP_MAX   = 8.0         # maximum signal amplitude (× noise std)
                                # → SNR varies across files

# Reverb
REVERB_DELAY_MS  = 5.0        # delay of reverb echo after pulse end (ms)
REVERB_DUR_MS    = 6.0        # duration of reverb echo (ms)
REVERB_AMP_RATIO = 0.35        # reverb amplitude as fraction of main pulse amplitude

# LFM start time — placed randomly in [START_MIN_MS, START_MAX_MS]
LFM_START_MIN_MS = 500.0       # earliest start (ms)
LFM_START_MAX_MS = 2500.0      # latest start   (ms)

# Random seed for reproducibility
RANDOM_SEED = 42

# Detection parameters (must match config.py EXTRACTION_* values)
RMS_WINDOW       = 1000
THRESHOLD_FACTOR = 2.0   # ← must match config.py EXTRACTION_THRESHOLD_FACTOR
PAD_SAMPLES      = 50
SAMPLE_RATE      = float(FS)

# ================================================================== #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Import extraction utilities
sys.path.insert(0, r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic")
from utils import compute_rms_envelope, estimate_noise_floor, detect_active_region


# --------------------------------------------------------------------------- #
# Signal generation helpers
# --------------------------------------------------------------------------- #

def make_lfm(duration_samples: int, f_start: float, f_end: float, fs: float) -> np.ndarray:
    """Generate a rectangular-envelope LFM (chirp) pulse — flat amplitude throughout."""
    t   = np.arange(duration_samples) / fs
    k   = (f_end - f_start) / t[-1]               # chirp rate (Hz/s)
    phi = 2.0 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    return np.sin(phi).astype(np.float32)          # no windowing — pure rectangular


def make_reverb(lfm: np.ndarray, amp_ratio: float, decay: float = 0.5) -> np.ndarray:
    """
    Simulate a simple reverb echo: a decayed, slightly blurred copy of the pulse.
    decay controls how fast the echo fades (higher = shorter tail).
    """
    n = len(lfm)
    env = np.exp(-decay * np.linspace(0, 4, n)).astype(np.float32)
    reverb = lfm * env * amp_ratio
    return reverb


def generate_file(rng: np.random.Generator, file_idx: int) -> dict:
    """
    Generate one synthetic trial recording.

    Returns a dict with:
        signal      : np.ndarray  (TOTAL_SAMPLES,)
        lfm_start   : int         sample index where LFM begins
        lfm_end     : int         sample index where LFM ends
        reverb_start: int         sample index where reverb begins
        reverb_end  : int         sample index where reverb ends
        signal_amp  : float       actual signal amplitude used
    """
    # Step 1 — Pure AWGN noise baseline
    signal = (rng.standard_normal(TOTAL_SAMPLES) * NOISE_STD).astype(np.float32)

    # Step 2 — LFM pulse
    lfm_dur_samples   = int(LFM_DURATION_MS / 1000.0 * FS)
    lfm_start_sample  = int(rng.uniform(
        LFM_START_MIN_MS / 1000.0 * FS,
        LFM_START_MAX_MS / 1000.0 * FS,
    ))
    lfm_end_sample    = lfm_start_sample + lfm_dur_samples

    signal_amp = float(rng.uniform(SIGNAL_AMP_MIN, SIGNAL_AMP_MAX)) * NOISE_STD
    lfm_pulse  = make_lfm(lfm_dur_samples, LFM_F_START, LFM_F_END, FS) * signal_amp

    # Add LFM into signal (also add extra AWGN on top of the pulse region)
    signal[lfm_start_sample:lfm_end_sample] += lfm_pulse
    signal[lfm_start_sample:lfm_end_sample] += (
        rng.standard_normal(lfm_dur_samples) * NOISE_STD
    ).astype(np.float32)

    # Step 3 — Reverb echo
    reverb_delay   = int(REVERB_DELAY_MS / 1000.0 * FS)
    reverb_dur     = int(REVERB_DUR_MS   / 1000.0 * FS)
    reverb_start   = lfm_end_sample + reverb_delay
    reverb_end     = reverb_start + reverb_dur

    if reverb_end <= TOTAL_SAMPLES:
        reverb_lfm   = make_lfm(reverb_dur, LFM_F_START, LFM_F_END, FS)
        reverb_pulse = make_reverb(reverb_lfm, REVERB_AMP_RATIO)
        reverb_pulse *= (signal_amp * 0.8)   # slightly quieter than main
        signal[reverb_start:reverb_end] += reverb_pulse
    else:
        reverb_start = reverb_end = -1       # off the edge — won't appear

    return {
        "signal"      : signal,
        "lfm_start"   : lfm_start_sample,
        "lfm_end"     : lfm_end_sample,
        "reverb_start": reverb_start,
        "reverb_end"  : reverb_end,
        "signal_amp"  : signal_amp,
        "snr_db"      : 20.0 * np.log10(signal_amp / NOISE_STD),
    }


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def save_dat(signal: np.ndarray, path: str) -> None:
    """Save 1-D float array as plain-text .dat (one value per line)."""
    with open(path, "w") as f:
        for v in signal:
            f.write(f"{v:.8e}\n")


# --------------------------------------------------------------------------- #
# Verification plot
# --------------------------------------------------------------------------- #

def verify_and_plot(
    signal: np.ndarray,
    truth: dict,
    file_path: str,
    plot_save_path: str,
) -> None:
    """
    Run the extractor on the generated file and plot:
      - True signal location (ground truth, shown as orange dashed lines)
      - Detected window (green lines)
      - RMS envelope with threshold
      - Masked output
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib not installed — skipping plot.")
        return

    # Run extractor
    rms         = compute_rms_envelope(signal, window=RMS_WINDOW)
    noise_floor = estimate_noise_floor(rms, percentile=20.0)
    det_start, det_end = detect_active_region(
        rms, noise_floor, THRESHOLD_FACTOR, PAD_SAMPLES, len(signal)
    )
    threshold   = THRESHOLD_FACTOR * noise_floor

    # Apply mask
    masked = np.zeros_like(signal)
    masked[det_start : det_end + 1] = signal[det_start : det_end + 1]

    # Time axes (ms)
    t = np.arange(len(signal)) / SAMPLE_RATE * 1000.0
    fname = os.path.basename(file_path)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(
        f"Verification — {fname}\n"
        f"True LFM: {truth['lfm_start']/SAMPLE_RATE*1000:.1f}–"
        f"{truth['lfm_end']/SAMPLE_RATE*1000:.1f} ms  |  "
        f"Signal amp: {truth['signal_amp']:.3f}  |  SNR: {truth['snr_db']:.1f} dB\n"
        f"Detected window: {det_start/SAMPLE_RATE*1000:.1f}–{det_end/SAMPLE_RATE*1000:.1f} ms  |  "
        f"Noise floor: {noise_floor:.5f}  Threshold: {threshold:.5f}",
        fontsize=10, fontweight="bold",
    )

    # Panel 1 — Raw signal with ground truth + detected window
    ax = axes[0]
    ax.plot(t, signal, color="#4c9be8", linewidth=0.3, label="Signal")
    # Ground truth (orange)
    ax.axvline(truth["lfm_start"] / SAMPLE_RATE * 1000,
               color="orange", linewidth=1.2, linestyle="--", label="True LFM start")
    ax.axvline(truth["lfm_end"]   / SAMPLE_RATE * 1000,
               color="orange", linewidth=1.2, linestyle="-.", label="True LFM end")
    if truth["reverb_start"] > 0:
        ax.axvspan(truth["reverb_start"] / SAMPLE_RATE * 1000,
                   truth["reverb_end"]   / SAMPLE_RATE * 1000,
                   alpha=0.15, color="purple", label="Reverb region")
    # Detected window (green)
    ax.axvline(det_start / SAMPLE_RATE * 1000, color="green",
               linewidth=1.2, linestyle="--", label=f"Detected start")
    ax.axvline(det_end   / SAMPLE_RATE * 1000, color="red",
               linewidth=1.2, linestyle="--", label=f"Detected end")
    ax.axvspan(det_start / SAMPLE_RATE * 1000,
               det_end   / SAMPLE_RATE * 1000,
               alpha=0.08, color="green")
    ax.set_title("Raw Signal — orange=truth, green/red=detected window, purple=reverb")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 2 — RMS envelope + threshold + noise floor
    ax = axes[1]
    ax.plot(t, rms, color="#e05c5c", linewidth=0.6, label="RMS envelope")
    ax.axhline(threshold,   color="orange", linewidth=1.2, linestyle="--",
               label=f"Threshold ({THRESHOLD_FACTOR:.1f}× noise floor = {threshold:.5f})")
    ax.axhline(noise_floor, color="gray",   linewidth=0.8, linestyle=":",
               label=f"Noise floor = {noise_floor:.5f}")
    ax.set_title("RMS Envelope")
    ax.set_ylabel("RMS")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Masked output
    ax = axes[2]
    ax.plot(t, masked, color="#6abf69", linewidth=0.4)
    ax.set_title("Extraction Output — noise regions zeroed")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (ms)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
    log.info("Plot saved → %s", plot_save_path)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,   exist_ok=True)

    log.info("=" * 60)
    log.info("Generating %d synthetic trial files", N_FILES)
    log.info("  Total samples : %d  (%.1f s @ %d Hz)", TOTAL_SAMPLES, TOTAL_SAMPLES / FS, FS)
    log.info("  LFM duration  : %.0f ms  (%d samples)", LFM_DURATION_MS, int(LFM_DURATION_MS / 1000 * FS))
    log.info("  Reverb delay  : %.0f ms  |  duration : %.0f ms", REVERB_DELAY_MS, REVERB_DUR_MS)
    log.info("  Output dir    : %s", OUTPUT_DIR)
    log.info("=" * 60)

    for i in range(1, N_FILES + 1):
        fname    = f"channel{i:03d}.dat"
        fpath    = os.path.join(OUTPUT_DIR, fname)
        plot_out = os.path.join(PLOT_DIR, f"verify_{i:03d}_{fname.replace('.dat', '')}.png")

        result = generate_file(rng, i)
        signal = result["signal"]

        log.info(
            "File %2d/%d  |  LFM @ %.0f–%.0f ms  |  SNR: %.1f dB  |  Saving…",
            i, N_FILES,
            result["lfm_start"] / FS * 1000,
            result["lfm_end"]   / FS * 1000,
            result["snr_db"],
        )
        save_dat(signal, fpath)
        log.info("  Saved → %s", fpath)

        log.info("  Running extractor + plotting…")
        verify_and_plot(signal, result, fpath, plot_out)
        log.info("  Done. Close the plot window to continue to file %d…\n", i + 1)

    log.info("=" * 60)
    log.info("All %d files generated and verified.", N_FILES)
    log.info("Files  → %s", OUTPUT_DIR)
    log.info("Plots  → %s", PLOT_DIR)
    log.info("")
    log.info("Update config.py to point TRIAL_SIGNAL_SUBDIR at:")
    log.info("  %s", os.path.dirname(OUTPUT_DIR))
    log.info("and TRIAL_SIGNAL_SUBDIR = %r", os.path.basename(OUTPUT_DIR).replace("Signal", "") )
    log.info("=" * 60)

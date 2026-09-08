"""
signal_extractor.py
--------------------
Pre-processing utility: detect the signal+noise active region in a trial
recording and zero-out the pure noise regions on either side.

What it does
------------
  1. Loads a raw trial .dat file.
  2. Estimates the noise floor from the quietest portion of the signal.
  3. Computes a sliding-window RMS envelope across the full signal.
  4. Detects where RMS exceeds (threshold × noise_floor_RMS).
  5. Keeps the active region + a configurable padding on each side
     (to include any reverb tail).
  6. Zeros out everything outside that window.
  7. Plots the result and saves it as a PNG.

Run from IDLE
-------------
  Open this file and press F5.
  Adjust the parameters in the CONFIG block below as needed.
  The detected window edges are shown on the plot — tweak THRESHOLD_FACTOR
  if the detection is cutting into the signal or keeping too much noise.
"""

import os
import logging
import sys
import numpy as np

# Import shared extraction functions from the project
sys.path.insert(0, r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic")
from utils import (
    load_dat_file,
    compute_rms_envelope,
    estimate_noise_floor,
    detect_active_region,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ================================================================== #
#                     USER CONFIGURATION                              #
#  Edit these values. Everything else is automatic.                  #
# ================================================================== #

# Path to the trial .dat file you want to process
INPUT_FILE = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Project\Signal\PRI_01\channel001.dat"

# Where to save the zeroed-out signal and the plot
OUTPUT_DIR = r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\inference_output"

# --- Detection parameters ---
# At 100 kHz, 400,000 samples = 4 seconds

# Sliding RMS window (samples).
# 1000 samples @ 100kHz = 10 ms — good for smoothing without blurring edges.
RMS_WINDOW = 1000

# How many times above the noise-floor RMS = "signal is present".
# Increase if noise spikes are falsely detected.
# Decrease if signal edges are being clipped.
THRESHOLD_FACTOR = 4.0

# Extra samples kept on BOTH sides of the detected active region.
# 5000 samples @ 100kHz = 50 ms — captures reverb tail.
PAD_SAMPLES = 5000

# Sampling rate (Hz) — used only for the time-axis label in the plot.
SAMPLE_RATE = 100_000.0

# If True, save the zeroed-out signal as a new .dat file alongside the PNG.
SAVE_DAT = True

# ================================================================== #


# --------------------------------------------------------------------------- #
# Masking helper  (apply_mask lives here; extraction functions are in utils.py)
# --------------------------------------------------------------------------- #

def apply_mask(signal: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return a copy of `signal` with everything outside [start, end] set to zero."""
    masked = np.zeros_like(signal)
    masked[start : end + 1] = signal[start : end + 1]
    return masked



# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def plot_extraction(
    signal: np.ndarray,
    masked: np.ndarray,
    rms: np.ndarray,
    start: int,
    end: int,
    noise_floor: float,
    threshold_factor: float,
    sample_rate: float,
    input_path: str,
    save_path: str,
) -> None:
    """
    3-panel plot:
      Panel 1 — Original signal with detected window highlighted
      Panel 2 — RMS envelope with threshold line
      Panel 3 — Masked (zeroed-outside) signal
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("matplotlib not installed. Cannot plot. Run: pip install matplotlib")
        return

    n = len(signal)
    t = np.arange(n) / sample_rate * 1000.0   # time in ms

    threshold_value = threshold_factor * noise_floor
    fname = os.path.basename(input_path)

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    fig.suptitle(
        f"Signal Extraction — {fname}\n"
        f"Detected window: sample {start}–{end}  "
        f"({start/sample_rate*1000:.1f}–{end/sample_rate*1000:.1f} ms)  |  "
        f"Threshold = {threshold_factor:.1f} × noise floor",
        fontsize=11, fontweight="bold",
    )

    # Panel 1 — Original signal with window shaded
    axes[0].plot(t, signal, color="#4c9be8", linewidth=0.5, label="Raw signal")
    axes[0].axvspan(
        start / sample_rate * 1000,
        end   / sample_rate * 1000,
        alpha=0.15, color="green", label="Kept region",
    )
    axes[0].axvline(start / sample_rate * 1000, color="green", linewidth=1.2, linestyle="--")
    axes[0].axvline(end   / sample_rate * 1000, color="red",   linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Original Signal  (green = kept, dashed lines = window edges)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — RMS envelope + threshold
    axes[1].plot(t, rms, color="#e05c5c", linewidth=0.8, label="RMS envelope")
    axes[1].axhline(
        threshold_value, color="orange", linewidth=1.2, linestyle="--",
        label=f"Threshold ({threshold_factor:.1f}× noise floor = {threshold_value:.4f})",
    )
    axes[1].axhline(
        noise_floor, color="gray", linewidth=0.8, linestyle=":",
        label=f"Noise floor RMS = {noise_floor:.4f}",
    )
    axes[1].set_ylabel("RMS")
    axes[1].set_title("Sliding-Window RMS Envelope")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Masked signal
    axes[2].plot(t, masked, color="#6abf69", linewidth=0.5)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_title("After Extraction (pure noise regions zeroed out)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Plot saved → %s", save_path)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Save masked .dat file
# --------------------------------------------------------------------------- #

def save_dat(signal: np.ndarray, path: str) -> None:
    """Save a 1-D float array as a plain-text .dat file (one value per line)."""
    with open(path, "w") as f:
        for v in signal:
            f.write(f"{v:.8e}\n")
    log.info("Masked signal saved → %s", path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # --- Import load utility from the project ---
    sys.path.insert(0, r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic")
    from utils import load_dat_file

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load
    log.info("Loading: %s", INPUT_FILE)
    signal = load_dat_file(INPUT_FILE)   # load at actual file length, no truncation
    log.info("Signal length: %d samples  (%.1f ms at %.0f Hz)",
             len(signal), len(signal) / SAMPLE_RATE * 1000, SAMPLE_RATE)

    # 2. RMS envelope
    rms = compute_rms_envelope(signal, window=RMS_WINDOW)

    # 3. Noise floor estimate
    noise_floor = estimate_noise_floor(rms, percentile=20.0)
    log.info("Estimated noise-floor RMS : %.6f", noise_floor)
    log.info("Detection threshold       : %.1f × %.6f = %.6f",
             THRESHOLD_FACTOR, noise_floor, THRESHOLD_FACTOR * noise_floor)

    # 4. Detect active window
    start, end = detect_active_region(
        rms, noise_floor, THRESHOLD_FACTOR, PAD_SAMPLES, len(signal)
    )
    log.info("Detected active region    : samples %d – %d  (%.1f – %.1f ms)",
             start, end,
             start / SAMPLE_RATE * 1000,
             end   / SAMPLE_RATE * 1000)
    log.info("Kept length               : %d samples  (%.1f ms)",
             end - start + 1, (end - start + 1) / SAMPLE_RATE * 1000)

    # 5. Apply mask
    masked = apply_mask(signal, start, end)

    # 6. Save .dat
    if SAVE_DAT:
        base = os.path.splitext(os.path.basename(INPUT_FILE))[0]
        dat_out = os.path.join(OUTPUT_DIR, f"{base}_extracted.dat")
        save_dat(masked, dat_out)

    # 7. Plot
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(OUTPUT_DIR, f"{ts}_extraction.png")

    plot_extraction(
        signal=signal,
        masked=masked,
        rms=rms,
        start=start,
        end=end,
        noise_floor=noise_floor,
        threshold_factor=THRESHOLD_FACTOR,
        sample_rate=SAMPLE_RATE,
        input_path=INPUT_FILE,
        save_path=png_path,
    )

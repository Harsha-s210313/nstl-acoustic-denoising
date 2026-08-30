"""
inference.py
------------
Inference script for the NSTL Acoustic Denoising Pipeline.

Usage
-----
Synthetic source (uses pretrained checkpoint by default):
    python inference.py --source synthetic

Trial source (requires fine-tuned checkpoint):
    python inference.py --source trial

Explicit checkpoint override:
    python inference.py --source synthetic --checkpoint models/unet_pretrained.pth
    python inference.py --source trial     --checkpoint models/unet_finetuned.pth

Optional: specify which file to run inference on:
    python inference.py --source synthetic --file data/synthetic/noisy/sig_001.dat
    python inference.py --source trial     --pri pri02 --channel 1

Outputs
-------
  - Console: SNR-proxy improvement (energy ratio before/after denoising)
  - File:    inference_output/<timestamp>_<source>_input.npy
             inference_output/<timestamp>_<source>_output.npy

Design choices
--------------
- Synthetic source → pretrained checkpoint  (default, always available after 2A)
- Trial source     → fine-tuned checkpoint  (requires 2B to have run)
- Source must be explicitly given; there is no opaque default that could
  silently apply a synthetic-only model to OOD trial data.
- The script clearly prints which source and checkpoint are being used.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch

from config import Config
from denoiser_finetune import _load_checkpoint
from unet import Configurable1DUNet
from utils import (
    ensure_dir,
    load_dat_file,
    normalize_signal,
    denormalize_signal,
    set_seed,
    to_tensor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# File resolution helpers
# --------------------------------------------------------------------------- #

def _resolve_synthetic_file(cfg: Config, explicit_file: str = None) -> str:
    """
    Return the path to a synthetic noisy file to use for inference.

    Priority:
    1. User-supplied --file argument.
    2. First file in the synthetic noisy/ directory (as a demo sample).

    Raises FileNotFoundError if neither source provides a valid file.
    """
    if explicit_file:
        if not os.path.isfile(explicit_file):
            raise FileNotFoundError(
                f"--file not found: {explicit_file!r}"
            )
        return explicit_file

    noisy_dir = os.path.join(
        cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_NOISY_SUBDIR
    )
    if not os.path.isdir(noisy_dir):
        raise FileNotFoundError(
            f"Synthetic noisy directory not found: {noisy_dir!r}. "
            "Run synthetic data generation first."
        )
    candidates = sorted(
        os.path.join(noisy_dir, f)
        for f in os.listdir(noisy_dir)
        if f.lower().endswith(".dat")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .dat files found in {noisy_dir!r}. "
            "Populate the synthetic noisy directory."
        )
    return candidates[0]


def _resolve_trial_file(cfg: Config, pri: str, channel: int) -> str:
    """
    Return the path to a specific trial signal file.

    Parameters
    ----------
    pri     : str  PRI folder name (e.g. "pri01")
    channel : int  Channel index (0-based)
    """
    fname = f"channel{channel}.dat"
    path = os.path.join(
        cfg.TRIAL_DATA_DIR, cfg.TRIAL_SIGNAL_SUBDIR, pri, fname
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trial file not found: {path!r}. "
            f"Check --pri and --channel arguments, and that TRIAL_DATA_DIR "
            f"({cfg.TRIAL_DATA_DIR!r}) is populated."
        )
    return path


# --------------------------------------------------------------------------- #
# Checkpoint resolution
# --------------------------------------------------------------------------- #

def _resolve_checkpoint(cfg: Config, source: str, explicit_ckpt: str = None) -> str:
    """
    Return the checkpoint path to use, with clear error messages.

    Logic:
    - If --checkpoint is given explicitly, use it (and validate it exists).
    - If source == "synthetic", use UNET_PRETRAINED_CHECKPOINT.
    - If source == "trial",     prefer UNET_FINETUNED_CHECKPOINT if it exists,
                                fall back to UNET_PRETRAINED_CHECKPOINT with a
                                prominent warning.
    """
    if explicit_ckpt:
        if not os.path.isfile(explicit_ckpt):
            raise FileNotFoundError(
                f"Explicitly specified checkpoint not found: {explicit_ckpt!r}"
            )
        return explicit_ckpt

    if source == "synthetic":
        ckpt = cfg.UNET_PRETRAINED_CHECKPOINT
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {ckpt!r}. "
                "Run Stage 2A pretraining first:\n"
                "    python denoiser_pretrain.py"
            )
        return ckpt

    # source == "trial"
    if os.path.isfile(cfg.UNET_FINETUNED_CHECKPOINT):
        return cfg.UNET_FINETUNED_CHECKPOINT
    else:
        log.warning(
            "\n"
            "⚠  Fine-tuned checkpoint not found at %r.\n"
            "   Falling back to pretrained checkpoint (%r).\n"
            "   NOTE: This model has NOT been fine-tuned on trial data.\n"
            "   Results may be suboptimal.  Run Stage 2B first:\n"
            "       python denoiser_finetune.py --stage finetune\n",
            cfg.UNET_FINETUNED_CHECKPOINT,
            cfg.UNET_PRETRAINED_CHECKPOINT,
        )
        ckpt = cfg.UNET_PRETRAINED_CHECKPOINT
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Neither fine-tuned ({cfg.UNET_FINETUNED_CHECKPOINT!r}) nor "
                f"pretrained ({cfg.UNET_PRETRAINED_CHECKPOINT!r}) checkpoint found. "
                "Run Stage 2A pretraining first."
            )
        return ckpt


# --------------------------------------------------------------------------- #
# Plotting helper
# --------------------------------------------------------------------------- #

def _plot_results(
    raw_signal: np.ndarray,
    denoised: np.ndarray,
    residual: np.ndarray,
    snr_db: float,
    source: str,
    input_path: str,
    save_path: str,
) -> None:
    """
    Display a 3-panel figure:
      Panel 1 — Noisy input signal
      Panel 2 — Denoised output signal
      Panel 3 — Residual (input − denoised)

    The figure is shown on screen (plt.show()) and saved as a PNG.

    Parameters
    ----------
    raw_signal  : 1-D array, original noisy input
    denoised    : 1-D array, U-Net output
    residual    : 1-D array, raw_signal − denoised
    snr_db      : float, SNR proxy value to display in title
    source      : "synthetic" or "trial"
    input_path  : path of the input file (shown in subtitle)
    save_path   : full path to save the PNG
    """
    try:
        import matplotlib
        matplotlib.use("TkAgg")          # works in IDLE; change to "Qt5Agg" if needed
        import matplotlib.pyplot as plt
    except ImportError:
        log.error(
            "matplotlib is not installed. Cannot plot results.\n"
            "Install it with: pip install matplotlib"
        )
        return

    samples = np.arange(len(raw_signal))
    fname   = os.path.basename(input_path)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        f"NSTL Acoustic Denoiser — {source.upper()} | {fname}\n"
        f"SNR proxy: {snr_db:.2f} dB",
        fontsize=12, fontweight="bold",
    )

    # Panel 1 — Noisy input
    axes[0].plot(samples, raw_signal, color="#e05c5c", linewidth=0.6)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Noisy Input")
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Denoised output
    axes[1].plot(samples, denoised, color="#4c9be8", linewidth=0.6)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Denoised Output")
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — Residual
    axes[2].plot(samples, residual, color="#6abf69", linewidth=0.6)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Sample index")
    axes[2].set_title("Residual (Input − Denoised)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save first, then show (show() blocks until the window is closed)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    log.info("Plot saved → %s", save_path)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Core inference function
# --------------------------------------------------------------------------- #

def run_inference(
    cfg: Config,
    source: str,
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
) -> None:
    """
    Load the model and run denoising on a single .dat file.

    Parameters
    ----------
    cfg            : Config
    source         : "synthetic" or "trial"
    checkpoint_path: path to the .pth checkpoint to use
    input_path     : path to the input .dat file
    output_dir     : directory to save output .npy arrays
    """
    ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 60)
    log.info("INFERENCE SUMMARY")
    log.info("  Source     : %s", source.upper())
    log.info("  Input file : %s", input_path)
    log.info("  Checkpoint : %s", checkpoint_path)
    log.info("  Device     : %s", device)
    log.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Load and pre-process input
    # ------------------------------------------------------------------ #
    raw_signal = load_dat_file(input_path, expected_length=cfg.SIGNAL_LENGTH)
    normalised, sig_mean, sig_std = normalize_signal(raw_signal)
    x = to_tensor(normalised).unsqueeze(0).to(device)   # (1, 1, L)

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    model = Configurable1DUNet(cfg).to(device)
    _load_checkpoint(checkpoint_path, model, cfg)
    model.eval()

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        y = model(x)                     # (1, 1, L)

    # ------------------------------------------------------------------ #
    # Post-process
    # ------------------------------------------------------------------ #
    denoised_norm = y.squeeze().cpu().numpy()             # (L,)
    denoised_raw = denormalize_signal(denoised_norm, sig_mean, sig_std)

    # Simple SNR-proxy: ratio of input vs. output energy relative to residual
    input_energy = float(np.mean(raw_signal ** 2))
    residual = raw_signal - denoised_raw
    residual_energy = float(np.mean(residual ** 2)) + 1e-12
    snr_proxy_db = 10.0 * np.log10(input_energy / residual_energy)

    log.info("Input signal  energy : %.6f", input_energy)
    log.info("Output signal energy : %.6f", float(np.mean(denoised_raw ** 2)))
    log.info("Residual energy      : %.6f", residual_energy)
    log.info("SNR proxy (dB)       : %.2f dB", snr_proxy_db)

    # ------------------------------------------------------------------ #
    # Plot — display on screen and save as PNG
    # ------------------------------------------------------------------ #
    ensure_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(output_dir, f"{ts}_{source}_result.png")

    _plot_results(
        raw_signal=raw_signal,
        denoised=denoised_raw,
        residual=residual,
        snr_db=snr_proxy_db,
        source=source,
        input_path=input_path,
        save_path=png_path,
    )
    log.info("Plot saved → %s", png_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "NSTL Acoustic Denoiser — Inference\n\n"
            "You MUST specify --source to avoid accidentally applying a\n"
            "synthetic-pretrained model to trial data or vice versa.\n\n"
            "Examples:\n"
            "  python inference.py --source synthetic\n"
            "  python inference.py --source trial --pri pri01 --channel 0\n"
            "  python inference.py --source synthetic --file data/synthetic/noisy/s1.dat\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "trial"],
        default=None,
        help=(
            "Data source for inference. "
            "'synthetic' uses the pretrained checkpoint. "
            "'trial' prefers the fine-tuned checkpoint (falls back to pretrained with warning)."
        ),
    )
    parser.add_argument(
        "--file",
        default=None,
        metavar="PATH",
        help="Explicit path to a .dat file. Overrides automatic file selection.",
    )
    parser.add_argument(
        "--pri",
        default=None,
        metavar="PRI",
        help=(
            "Trial PRI folder (e.g. pri01). Used when --source=trial and --file is not given. "
            "Defaults to Config.INFERENCE_TRIAL_PRI."
        ),
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Trial channel index (0-based). Used when --source=trial and --file is not given. "
            "Defaults to Config.INFERENCE_TRIAL_CHANNEL."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Explicit checkpoint path.  Overrides automatic checkpoint selection.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to save output arrays.  Defaults to Config.INFERENCE_OUTPUT_DIR.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    # ------------------------------------------------------------------ #
    # Source must be explicit
    # ------------------------------------------------------------------ #
    if args.source is None:
        log.info(
            "No --source specified.  Using default from Config: '%s'.",
            cfg.INFERENCE_DEFAULT_SOURCE,
        )
        source = cfg.INFERENCE_DEFAULT_SOURCE
    else:
        source = args.source

    # ------------------------------------------------------------------ #
    # Resolve checkpoint
    # ------------------------------------------------------------------ #
    checkpoint_path = _resolve_checkpoint(cfg, source, args.checkpoint)

    # ------------------------------------------------------------------ #
    # Resolve input file
    # ------------------------------------------------------------------ #
    if args.file:
        input_path = args.file
        if not os.path.isfile(input_path):
            log.error("--file not found: %s", input_path)
            sys.exit(1)
    elif source == "synthetic":
        input_path = _resolve_synthetic_file(cfg)
    else:  # trial
        pri = args.pri if args.pri is not None else cfg.INFERENCE_TRIAL_PRI
        channel = args.channel if args.channel is not None else cfg.INFERENCE_TRIAL_CHANNEL
        input_path = _resolve_trial_file(cfg, pri, channel)

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    output_dir = args.output_dir if args.output_dir else cfg.INFERENCE_OUTPUT_DIR

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    run_inference(cfg, source, checkpoint_path, input_path, output_dir)

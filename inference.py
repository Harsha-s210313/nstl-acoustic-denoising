"""
Inference engine for underwater acoustic signal detection and denoising.
Processes synthetic or trial recordings, performs active-region extraction,
applies Stage 1 Detector and Stage 2 U-Net Denoiser, computes SNR metrics,
and exports text .dat files and headless visual diagnostic plots.
"""

import os
import sys
import glob
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

try:
    from .config import Config, default_config
    from .utils import (
        load_dat_file,
        save_dat_file,
        normalize_signal,
        denormalize_signal,
        extract_active_region,
        extract_noise_reference,
        apply_bandpass_filter,
        ensure_dir,
        to_tensor,
    )
    from .detector import SignalDetectorCNN
    from .unet import Configurable1DUNet
except (ImportError, ValueError):
    from config import Config, default_config
    from utils import (
        load_dat_file,
        save_dat_file,
        normalize_signal,
        denormalize_signal,
        extract_active_region,
        extract_noise_reference,
        apply_bandpass_filter,
        ensure_dir,
        to_tensor,
    )
    from detector import SignalDetectorCNN
    from unet import Configurable1DUNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels: 10 * log10(P_sig / P_noise).
    """
    p_sig = float(np.mean(signal ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise < 1e-12:
        return 50.0  # Cap at high SNR
    if p_sig < 1e-12:
        return -50.0
    return float(10.0 * np.log10(p_sig / p_noise))


def run_pipeline(
    source: str = "synthetic",
    pri: Optional[str] = None,
    channel: Optional[str] = None,
    model_type: Optional[str] = None,
    checkpoint: Optional[str] = None,
    detector_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    cfg: Optional[Config] = None,
    save_plot: bool = True,
    show_plot: bool = False,
) -> Dict[str, Any]:
    """
    Execute full detection + denoising inference pipeline.
    """
    config = cfg or default_config
    config.validate()

    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    # 1. Resolve Input File Path
    clean_path = None
    is_trial = False

    if source.lower() == "synthetic":
        noisy_dir = os.path.join(config.SYNTHETIC_DATA_DIR, config.SYNTHETIC_NOISY_SUBDIR)
        if not os.path.isdir(noisy_dir):
            for alt in [
                os.path.join(config.SYNTHETIC_DATA_DIR, "synthetic", "noisy"),
                os.path.join(config.SYNTHETIC_DATA_DIR, "noisy"),
            ]:
                if os.path.isdir(alt):
                    noisy_dir = alt
                    break

        clean_dir = os.path.join(config.SYNTHETIC_DATA_DIR, config.SYNTHETIC_CLEAN_SUBDIR)
        if not os.path.isdir(clean_dir):
            for alt in [
                os.path.join(config.SYNTHETIC_DATA_DIR, "synthetic", "clean"),
                os.path.join(config.SYNTHETIC_DATA_DIR, "clean"),
            ]:
                if os.path.isdir(alt):
                    clean_dir = alt
                    break

        available_files = sorted(glob.glob(os.path.join(noisy_dir, "*.dat")))
        if not available_files:
            raise FileNotFoundError(f"No synthetic noisy files found in {noisy_dir}")
        input_path = available_files[0]
        c_candidate = os.path.join(clean_dir, os.path.basename(input_path))
        if os.path.isfile(c_candidate):
            clean_path = c_candidate
    elif source.lower() == "trial":
        is_trial = True
        pri_val = pri if pri is not None else config.INFERENCE_TRIAL_PRI
        ch_val = channel or config.INFERENCE_TRIAL_CHANNEL

        # Check candidate paths in priority order
        candidates = []
        if pri_val and config.TRIAL_SIGNAL_SUBDIR:
            candidates.append(os.path.join(config.TRIAL_DATA_DIR, config.TRIAL_SIGNAL_SUBDIR, pri_val, ch_val))
        if pri_val:
            candidates.append(os.path.join(config.TRIAL_DATA_DIR, pri_val, ch_val))
        if config.TRIAL_SIGNAL_SUBDIR:
            candidates.append(os.path.join(config.TRIAL_DATA_DIR, config.TRIAL_SIGNAL_SUBDIR, ch_val))
        candidates.append(os.path.join(config.TRIAL_DATA_DIR, ch_val))

        input_path = None
        for cand in candidates:
            if os.path.isfile(cand):
                input_path = cand
                break

        if input_path is None:
            # Search recursively for matching channel name or any dat file in TRIAL_DATA_DIR
            pattern = os.path.join(config.TRIAL_DATA_DIR, "**", ch_val)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                input_path = matches[0]
            else:
                all_dats = sorted(glob.glob(os.path.join(config.TRIAL_DATA_DIR, "**", "*.dat"), recursive=True))
                if all_dats:
                    input_path = all_dats[0]
                    logger.warning("Requested trial file '%s' not found. Falling back to %s", ch_val, input_path)
                else:
                    raise FileNotFoundError(f"No trial data files found under {config.TRIAL_DATA_DIR}")
    else:
        # User supplied explicit path
        input_path = source
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        raw_check = load_dat_file(input_path)
        is_trial = len(raw_check) < 22000

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    logger.info("Running inference on: %s (is_trial=%s)", input_path, is_trial)

    # 2. Preprocess Signal
    raw_signal = load_dat_file(input_path)
    clean_signal = None

    if is_trial:
        ext_dict = extract_active_region(raw_signal, config, target_length=config.TRIAL_SIGNAL_LENGTH)
        eval_signal = ext_dict["active"]
        noise_ref = ext_dict["noise"]
        active_bounds = (ext_dict["start"], ext_dict["end"])
    else:
        if len(raw_signal) != config.SIGNAL_LENGTH:
            eval_signal = load_dat_file(input_path, expected_length=config.SIGNAL_LENGTH)
        else:
            eval_signal = raw_signal
        if clean_path and os.path.isfile(clean_path):
            clean_signal = load_dat_file(clean_path, expected_length=config.SIGNAL_LENGTH)
        noise_ref = None
        active_bounds = (0, len(eval_signal))

    # 3. Stage 1: Detector Evaluation
    det_ckpt_path = detector_checkpoint or config.DETECTOR_CHECKPOINT
    detector_verdict = "Detector checkpoint not loaded"
    detector_confidence = 0.0

    if os.path.isfile(det_ckpt_path):
        try:
            det_model = SignalDetectorCNN(cfg=config).to(dev)
            det_ckpt = torch.load(det_ckpt_path, map_location=dev)
            det_model.load_state_dict(det_ckpt["model_state_dict"])
            det_model.eval()

            # Normalize for detector input
            det_norm, _, _ = normalize_signal(eval_signal)
            det_tensor = to_tensor(det_norm).to(dev)
            preds, probs = det_model.predict(det_tensor)
            pred_class = int(preds[0].item())
            prob_signal = float(probs[0, 1].item())
            detector_confidence = prob_signal
            verdict_text = "Pulse Present" if pred_class == 1 else "Noise Only"
            detector_verdict = f"{verdict_text} (Conf: {prob_signal * 100.0:.1f}%)"
            logger.info("Detector Stage: %s", detector_verdict)
        except Exception as e:
            logger.warning("Failed evaluating detector: %s", e)
    else:
        logger.info("Detector checkpoint not found at %s. Skipping Stage 1.", det_ckpt_path)

    if checkpoint:
        unet_ckpt_path = checkpoint
    else:
        if model_type is None:
            if is_trial and os.path.isfile(config.UNET_FINETUNED_CHECKPOINT):
                model_type = "finetuned"
            else:
                model_type = "pretrained"

        if model_type.lower() == "finetuned":
            unet_ckpt_path = config.UNET_FINETUNED_CHECKPOINT
        else:
            unet_ckpt_path = config.UNET_PRETRAINED_CHECKPOINT

    if not os.path.isfile(unet_ckpt_path):
        raise FileNotFoundError(
            f"Denoiser checkpoint not found at: {unet_ckpt_path}. "
            "Please train or pretrain the model first."
        )

    logger.info("Loading U-Net denoiser weights from %s", unet_ckpt_path)
    denoiser = Configurable1DUNet(cfg=config).to(dev)
    ckpt = torch.load(unet_ckpt_path, map_location=dev)
    denoiser.load_state_dict(ckpt["model_state_dict"])
    denoiser.eval()

    # Normalize input
    norm_input, mean_val, std_val = normalize_signal(eval_signal)
    input_tensor = to_tensor(norm_input).to(dev)

    with torch.no_grad():
        output_tensor = denoiser(input_tensor)

    denoised_norm = output_tensor.squeeze().cpu().numpy()
    denoised_signal = denormalize_signal(denoised_norm, mean_val, std_val)

    # Apply zero-phase bandpass cleanup to eliminate out-of-band piecewise ReLU convolution hash
    if getattr(config, "INFERENCE_CLEANUP_BPF", True):
        fs_sig = config.TRIAL_SAMPLE_RATE if is_trial else config.SAMPLE_RATE
        if is_trial:
            denoised_signal = apply_bandpass_filter(denoised_signal, fs=fs_sig, lowcut=2500.0, highcut=6500.0)
        else:
            denoised_signal = apply_bandpass_filter(denoised_signal, fs=fs_sig, lowcut=15000.0, highcut=30000.0)

    # 5. Compute SNR Metrics
    snr_improvement = 0.0
    if clean_signal is not None:
        snr_in = calculate_snr_db(clean_signal, eval_signal - clean_signal)
        snr_out = calculate_snr_db(clean_signal, denoised_signal - clean_signal)
        snr_improvement = snr_out - snr_in
        logger.info("Ground Truth Evaluation: Input SNR=%.2f dB, Output SNR=%.2f dB, Gain=%+.2f dB", snr_in, snr_out, snr_improvement)
    else:
        # Unsupervised estimate: compare signal-to-noise ratio before and after denoising
        # Use background noise reference from the quiet part of the recording
        start, end = active_bounds
        act_len = max(1, end - start)
        sig_active_in = eval_signal[:act_len] if len(eval_signal) >= act_len else eval_signal
        sig_active_out = denoised_signal[:act_len] if len(denoised_signal) >= act_len else denoised_signal

        if noise_ref is None or len(noise_ref) == 0:
            noise_ref = extract_noise_reference(raw_signal, min(1000, len(raw_signal) // 4))

        ref_len = min(len(noise_ref), len(eval_signal))
        noise_slice = noise_ref[:ref_len]
        p_noise_in = float(np.mean(noise_slice ** 2)) + 1e-12

        # Denoise noise slice to measure residual noise floor
        norm_n, m_n, s_n = normalize_signal(noise_slice)
        with torch.no_grad():
            den_n_norm = denoiser(to_tensor(norm_n).to(dev)).squeeze().cpu().numpy()
        den_n = denormalize_signal(den_n_norm, m_n, s_n)
        p_noise_out = float(np.mean(den_n ** 2)) + 1e-12

        p_sig_in = float(np.mean(sig_active_in ** 2))
        p_sig_out = float(np.mean(sig_active_out ** 2))

        est_snr_in = float(10.0 * np.log10(max(p_sig_in - p_noise_in, 1e-8) / p_noise_in))
        est_snr_out = float(10.0 * np.log10(max(p_sig_out - p_noise_out, 1e-8) / p_noise_out))
        snr_improvement = est_snr_out - est_snr_in
        logger.info("Estimated SNR Gain: %+.2f dB (In: %.2f dB, Out: %.2f dB)", snr_improvement, est_snr_in, est_snr_out)

    # 6. Save Denoised Output as Plain-Text .dat and metadata JSON sidecar
    ensure_dir(config.INFERENCE_OUTPUT_DIR)
    out_dat_path = os.path.join(config.INFERENCE_OUTPUT_DIR, f"{base_name}_denoised.dat")
    out_meta_path = os.path.join(config.INFERENCE_OUTPUT_DIR, f"{base_name}_meta.json")
    start, end = active_bounds
    act_len = end - start
    if config.INFERENCE_RECONSTRUCT_FULL and len(raw_signal) > len(eval_signal) and act_len > 0:
        out_signal = np.zeros_like(raw_signal)
        valid_act_len = min(act_len, len(denoised_signal))
        out_signal[start : start + valid_act_len] = denoised_signal[:valid_act_len]
        reconstructed_full = True
    else:
        out_signal = denoised_signal[:len(raw_signal)]
        reconstructed_full = False
    save_dat_file(out_dat_path, out_signal)
    logger.info("Saved denoised output to: %s", out_dat_path)

    import json
    meta_info = {
        "base_name": base_name,
        "input_path": input_path,
        "output_dat": out_dat_path,
        "is_trial": is_trial,
        "crop_start": int(start),
        "crop_end": int(end),
        "crop_length": int(act_len),
        "raw_length": int(len(raw_signal)),
        "eval_length": int(len(eval_signal)),
        "reconstructed_full": reconstructed_full,
        "detector_verdict": detector_verdict,
        "detector_confidence": detector_confidence,
        "snr_improvement_db": float(snr_improvement),
    }
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    logger.info("Saved metadata sidecar to: %s", out_meta_path)

    # 7. Generate Visual Diagnostic Plot (Lazy matplotlib import, Agg backend)
    plot_path = None
    if save_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scipy.signal import welch

            plot_path = os.path.join(config.INFERENCE_OUTPUT_DIR, f"{base_name}_analysis.png")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

            # Subplot 1: Time Series
            fs = config.TRIAL_SAMPLE_RATE if is_trial else config.SAMPLE_RATE
            start_sample = active_bounds[0] if (is_trial and len(raw_signal) > len(eval_signal)) else 0
            t = (start_sample + np.arange(len(eval_signal))) / fs * 1000.0  # ms
            ax1.plot(t, eval_signal, color="#b0bec5", alpha=0.8, label="Input Noisy")
            if clean_signal is not None:
                ax1.plot(t, clean_signal, color="#2e7d32", alpha=0.5, linestyle="--", label="Ground Truth Clean")
            ax1.plot(t, denoised_signal, color="#1565c0", linewidth=1.2, label=f"Denoised ({model_type})")
            ax1.set_title(f"Acoustic Time Series [{base_name}] | Detector: {detector_verdict}", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.legend(loc="upper right")

            # Subplot 2: Power Spectral Density (PSD)
            nperseg = min(512, len(eval_signal))
            f_noisy, psd_noisy = welch(eval_signal, fs=fs, nperseg=nperseg)
            f_denoised, psd_denoised = welch(denoised_signal, fs=fs, nperseg=nperseg)

            ax2.semilogy(f_noisy / 1000.0, psd_noisy, color="#b0bec5", label="Noisy PSD")
            if clean_signal is not None:
                f_clean, psd_clean = welch(clean_signal, fs=fs, nperseg=nperseg)
                ax2.semilogy(f_clean / 1000.0, psd_clean, color="#2e7d32", linestyle="--", label="Clean PSD")
            ax2.semilogy(f_denoised / 1000.0, psd_denoised, color="#1565c0", label="Denoised PSD")
            ax2.set_title(f"Power Spectral Density (Welch) | Estimated SNR Gain: {snr_improvement:+.2f} dB", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Frequency (kHz)")
            ax2.set_ylabel("PSD (V^2 / Hz)")
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper right")

            plt.tight_layout()
            try:
                plt.savefig(plot_path, dpi=150)
            except OSError:
                plot_path = os.path.join(config.INFERENCE_OUTPUT_DIR, f"{base_name}_analysis_latest.png")
                plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info("Saved diagnostic plot to: %s", plot_path)

            if show_plot and plot_path and os.path.isfile(plot_path):
                if sys.platform.startswith("win"):
                    try:
                        os.startfile(plot_path)
                    except Exception as ex:
                        logger.debug("Could not open plot in image viewer: %s", ex)
        except Exception as e:
            logger.warning("Could not generate diagnostic plot: %s", e)

    return {
        "input_path": input_path,
        "output_dat": out_dat_path,
        "output_plot": plot_path,
        "output_meta": out_meta_path,
        "detector_verdict": detector_verdict,
        "detector_confidence": detector_confidence,
        "snr_improvement": snr_improvement,
        "is_trial": is_trial,
        "active_bounds": active_bounds,
    }


def main():
    parser = argparse.ArgumentParser(description="Acoustic Pipeline Detection and Denoising Inference")
    parser.add_argument("--source", type=str, default="synthetic", help="Source: 'synthetic', 'trial', or direct file path")
    parser.add_argument("--pri", type=str, default=None, help="Trial PRI folder (e.g. 'PRI_01')")
    parser.add_argument("--channel", type=str, default=None, help="Trial channel file (e.g. 'channel001.dat')")
    parser.add_argument("--model", type=str, choices=["pretrained", "finetuned"], default=None, help="U-Net checkpoint stage")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit denoiser checkpoint path")
    parser.add_argument("--detector-checkpoint", type=str, default=None, help="Explicit detector checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--show-plot", action="store_true", help="Open generated diagnostic plot in viewer")
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        pri=args.pri,
        channel=args.channel,
        model_type=args.model,
        checkpoint=args.checkpoint,
        detector_checkpoint=args.detector_checkpoint,
        device=args.device,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()

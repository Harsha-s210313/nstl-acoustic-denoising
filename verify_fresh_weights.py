"""
Verification script for fresh model weights on real sea trial files.
Evaluates:
- Matched filter correlation peak of raw vs denoised
- Ratio of denoised / raw peak
- Alignment shift (lag)
- SNR gain (dB)
"""

import os
import glob
import numpy as np
import torch
from config import Config
from utils import (
    load_dat_file,
    normalize_signal,
    denormalize_signal,
    to_tensor,
    apply_bandpass_filter,
    extract_active_region,
)
from unet import Configurable1DUNet


def verify():
    cfg = Config()
    trial_files = sorted(glob.glob(r"C:\Users\HARSHA\Documents\pri\channel*.dat"))
    print(f"Found {len(trial_files)} trial files in C:\\Users\\HARSHA\\Documents\\pri")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    unet = Configurable1DUNet(cfg=cfg).to(dev)
    ckpt = torch.load(cfg.UNET_FINETUNED_CHECKPOINT, map_location=dev)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()

    # Construct standard LFM chirp replica
    fs = 17800.0
    dur = 0.113
    p_len = int(round(dur * fs))  # 2011 samples
    t = np.arange(p_len, dtype=np.float32) / fs
    f1, f2 = 3200.0, 5700.0
    k = (f2 - f1) / dur
    replica = (np.sin(2.0 * np.pi * (f1 * t + 0.5 * k * (t ** 2))) * np.hanning(p_len)).astype(np.float32)
    replica_norm = replica / (np.sqrt(np.sum(replica ** 2)) + 1e-12)

    print("\n" + "=" * 80)
    print(f"{'Channel':<16} | {'Raw Peak':<10} | {'Den Peak':<10} | {'Peak Ratio':<10} | {'Lag (samples)':<13} | {'SNR Gain (dB)'}")
    print("=" * 80)

    for fpath in trial_files:
        fname = os.path.basename(fpath)
        raw = load_dat_file(fpath)

        ext = extract_active_region(raw, cfg, target_length=cfg.TRIAL_SIGNAL_LENGTH)
        eval_sig = ext["active"]
        noise_ref = ext["noise"]

        norm_in, m_val, s_val = normalize_signal(eval_sig)
        with torch.no_grad():
            out_tensor = unet(to_tensor(norm_in).to(dev))
        den_norm = out_tensor.squeeze().cpu().numpy()
        denoised = denormalize_signal(den_norm, m_val, s_val)
        denoised = apply_bandpass_filter(denoised, fs=fs, lowcut=2500.0, highcut=6500.0)

        # Matched filter correlations
        c_raw = np.correlate(eval_sig, replica_norm, mode="full")
        c_den = np.correlate(denoised, replica_norm, mode="full")

        peak_raw = float(np.max(c_raw))
        peak_den = float(np.max(c_den))
        ratio = peak_den / max(peak_raw, 1e-6)

        lag = int(np.argmax(c_raw)) - int(np.argmax(c_den))

        # SNR calculation
        p_n_in = float(np.mean(noise_ref[:1000] ** 2)) + 1e-12
        p_s_in = float(np.mean(eval_sig[:2500] ** 2))
        snr_in = 10.0 * np.log10(max(p_s_in - p_n_in, 1e-6) / p_n_in)

        norm_n, mn_n, sn_n = normalize_signal(noise_ref[:1000])
        with torch.no_grad():
            out_n = unet(to_tensor(norm_n).to(dev)).squeeze().cpu().numpy()
        den_n = apply_bandpass_filter(denormalize_signal(out_n, mn_n, sn_n), fs=fs, lowcut=2500.0, highcut=6500.0)
        p_n_out = float(np.mean(den_n ** 2)) + 1e-12
        p_s_out = float(np.mean(denoised[:2500] ** 2))
        snr_out = 10.0 * np.log10(max(p_s_out - p_n_out, 1e-6) / p_n_out)
        snr_gain = snr_out - snr_in

        print(f"{fname:<16} | {peak_raw:<10.1f} | {peak_den:<10.1f} | {ratio:<10.2f} | {lag:<13d} | {snr_gain:+.2f} dB")

    print("=" * 80)


if __name__ == "__main__":
    verify()

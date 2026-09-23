"""
Better 3-panel visualization:
  Panel 1 — Noisy input (0-200ms), full scale
  Panel 2 — Denoised output, auto-scaled so you can see structure
  Panel 3 — Removed noise = noisy - denoised  (what the model suppressed)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic")

from config import Config
from unet import Configurable1DUNet
from denoiser_finetune import _load_checkpoint
from utils import load_dat_file, normalize_signal, set_seed

def to_tensor(x):
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)

cfg    = Config()
sr     = cfg.TRIAL_SAMPLE_RATE   # 17,800 Hz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(cfg.RANDOM_SEED)

# Load pretrained model
model = Configurable1DUNet(cfg).to(device)
_load_checkpoint(cfg.UNET_PRETRAINED_CHECKPOINT, model, cfg)
model.eval()

# Load real trial file
trial_file = r"C:\Users\HARSHA\Downloads\TransferNow-20260922dMEzqqdd\pri\channel001.dat"
full_signal = load_dat_file(trial_file)
n = len(full_signal)

# Manual window 0-200ms
start = 0
end   = int(200.0 * sr / 1000.0)
end   = min(end, n - 1)

window     = full_signal[start : end + 1]
target_len = cfg.TRIAL_SIGNAL_LENGTH
active     = np.pad(window, (0, max(0, target_len - len(window))), mode="constant") if len(window) < target_len else window[:target_len].copy()

active_norm, act_mean, act_std = normalize_signal(active)
x = to_tensor(active_norm).unsqueeze(0).to(device)

with torch.no_grad():
    y = model(x)

denoised_norm   = y.squeeze().cpu().numpy()[:len(active)]
denoised_active = denoised_norm * act_std + act_mean

active_len     = min(end - start + 1, target_len)
active_input   = full_signal[start : start + active_len]
denoised_out   = denoised_active[:active_len]
removed_noise  = active_input - denoised_out

t_zoom = np.arange(active_len) / sr * 1000.0   # ms

snr_db = 10.0 * np.log10(np.mean(active_input**2) / (np.mean(removed_noise**2) + 1e-12))
print(f"SNR proxy: {snr_db:.2f} dB")
print(f"Input    amplitude: {active_input.min():.1f} to {active_input.max():.1f}")
print(f"Denoised amplitude: {denoised_out.min():.1f}   to {denoised_out.max():.1f}")
print(f"Removed  amplitude: {removed_noise.min():.1f} to {removed_noise.max():.1f}")

# ── 3-panel plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(
    f"NSTL Trial Inference — channel001.dat  [PRETRAINED]\n"
    f"0-200 ms (LFM signal region)  |  SNR proxy: {snr_db:.2f} dB",
    fontsize=11, fontweight="bold",
)

# Panel 1: Noisy input full scale
axes[0].plot(t_zoom, active_input, color="#4c9be8", lw=0.6, alpha=0.8)
axes[0].set_title("Noisy Input  (full amplitude scale — dominated by impulsive noise bursts)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True, alpha=0.3)

# Panel 2: Denoised output — auto-scaled to show signal structure
axes[1].plot(t_zoom, denoised_out, color="#e05c5c", lw=0.8)
axes[1].set_title("Denoised Output  (auto-scaled — noise bursts removed, signal structure visible)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)

# Overlay them on Panel 2 with clipped y-axis to compare
perc = np.percentile(np.abs(denoised_out), 99)
ylim = max(perc * 2, 1.0)
axes[1].set_ylim(-ylim * 3, ylim * 3)

# Panel 3: Noise removed by model
axes[2].plot(t_zoom, removed_noise, color="#888888", lw=0.5, alpha=0.7)
axes[2].set_title("Removed Noise  (noisy - denoised) — impulsive spikes the model suppressed")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Time (ms)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("inference_output", exist_ok=True)
out = "inference_output/pretrained_REAL_3panel.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"PLOT SAVED -> {out}")

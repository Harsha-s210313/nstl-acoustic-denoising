# run_inference.py
# ----------------
# Open in IDLE and press F5.
#
# CURRENT SETUP:
#   - SOURCE = "trial"
#   - Uses PRETRAINED checkpoint (finetuned was deleted — trained with broken loss)
#   - Points at your real trial file (channel001.dat)
#   - Manual window: 0–200 ms  (the LFM signal is in this region)
#     Adjust MANUAL_END_MS up/down if the signal looks cut off in the plot.

import os
from config import Config
from inference import run_inference
from utils import set_seed

# ================================================================
# CONFIGURE HERE
# ================================================================

SOURCE = "trial"

# Your real trial file
EXPLICIT_FILE = r"C:\Users\HARSHA\Downloads\TransferNow-20260922dMEzqqdd\pri\channel001.dat"

# Pretrained checkpoint (not the finetuned one — that had a broken loss)
CHECKPOINT = r"models\unet_pretrained.pth"

# Manual active window in milliseconds.
# Auto-detection fails on this file because the background noise is bursty.
# The LFM signal is visible in the 0–200 ms range.
# Tweak MANUAL_END_MS if the denoised signal looks truncated.
MANUAL_START_MS = 0.0     # ms
MANUAL_END_MS   = 200.0   # ms — increase to 250 or 300 if signal is cut off

# ================================================================

cfg = Config()
set_seed(cfg.RANDOM_SEED)

print(f"\n[run_inference] Source     : {SOURCE.upper()}")
print(f"[run_inference] File       : {EXPLICIT_FILE}")
print(f"[run_inference] Checkpoint : {CHECKPOINT}")
print(f"[run_inference] Window     : {MANUAL_START_MS:.1f} - {MANUAL_END_MS:.1f} ms (manual)\n")

run_inference(
    cfg,
    SOURCE,
    CHECKPOINT,
    EXPLICIT_FILE,
    cfg.INFERENCE_OUTPUT_DIR,
    manual_start_ms=MANUAL_START_MS,
    manual_end_ms=MANUAL_END_MS,
)

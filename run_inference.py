# run_inference.py
# ----------------
# Open this file in IDLE and press F5 to run inference.
# No command-line arguments needed.
#
# SOURCE = "synthetic"  →  uses pretrained checkpoint  (verifies model works)
# SOURCE = "trial"      →  uses fine-tuned checkpoint  (real pipeline output)
#
# For trial source:
#   The full 400,000-sample file is loaded, the active signal+noise region is
#   detected automatically, denoised, then placed back into the full recording
#   with noise regions zeroed. A 4-panel plot is shown and saved as PNG.

import os
from config import Config
from denoiser_finetune import _load_checkpoint
from inference import run_inference, _resolve_checkpoint
from utils import set_seed

# ================================================================
# *** CONFIGURE YOUR INFERENCE HERE ***
# ================================================================

SOURCE = "trial"       # "synthetic"  or  "trial"

# --- Used when SOURCE = "trial" and EXPLICIT_FILE = None ---
# These select which generated file to run inference on.
# PRI folder = "PRI_TEST", channel = 1 → channel001.dat
TRIAL_PRI     = "PRI_TEST"
TRIAL_CHANNEL = 1          # 1-based: 1 = channel001.dat, 2 = channel002.dat, etc.

# Set an explicit file path, OR set to None to auto-resolve via TRIAL_PRI/TRIAL_CHANNEL.
EXPLICIT_FILE = None
# Example override:
# EXPLICIT_FILE = r"C:\Users\HARSHA\.gemini\antigravity\scratch\nstl_acoustic\data\trial_data\Signal\PRI_TEST\channel003.dat"

# ================================================================

cfg = Config()
set_seed(cfg.RANDOM_SEED)

# Resolve checkpoint
checkpoint_path = _resolve_checkpoint(cfg, SOURCE)

# Resolve input file
if EXPLICIT_FILE:
    input_path = EXPLICIT_FILE
elif SOURCE == "synthetic":
    # Pick any .dat file from the synthetic noisy folder
    noisy_dir = cfg.SYNTHETIC_NOISY_SUBDIR
    candidates = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".dat"))
    if not candidates:
        raise FileNotFoundError(f"No .dat files found in {noisy_dir!r}")
    input_path = os.path.join(noisy_dir, candidates[0])
else:
    # Build path from PRI + channel number
    signal_root = cfg.TRIAL_SIGNAL_SUBDIR
    chan_file   = f"channel{TRIAL_CHANNEL:03d}.dat"
    input_path  = os.path.join(signal_root, TRIAL_PRI, chan_file)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Trial file not found: {input_path!r}\n"
            f"Check TRIAL_PRI={TRIAL_PRI!r} and TRIAL_CHANNEL={TRIAL_CHANNEL}."
        )

print(f"\n[run_inference] Source   : {SOURCE.upper()}")
print(f"[run_inference] File     : {input_path}")
print(f"[run_inference] Checkpoint: {checkpoint_path}\n")

run_inference(cfg, SOURCE, checkpoint_path, input_path, cfg.INFERENCE_OUTPUT_DIR)

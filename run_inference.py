# run_inference.py
# ----------------
# Open this file in IDLE and press F5 to run inference.
# No command-line arguments needed.
#
# Set SOURCE to either:
#   "synthetic"  →  uses pretrained checkpoint  (use this first to verify)
#   "trial"      →  uses fine-tuned checkpoint  (use after fine-tuning)

from config import Config
from denoiser_finetune import _load_checkpoint
from inference import run_inference, _resolve_checkpoint, _resolve_synthetic_file, _resolve_trial_file
from utils import set_seed

# ================================================================
# *** CONFIGURE YOUR INFERENCE HERE ***
# ================================================================

SOURCE = "synthetic"        # <-- CHANGE to "trial" when ready

# Only used when SOURCE = "trial"
TRIAL_PRI     = "pri01"     # <-- CHANGE to your PRI folder name
TRIAL_CHANNEL = 0           # <-- CHANGE to your channel number (0-based)

# Leave as None to use the first available file automatically,
# or set an explicit path, e.g.:
EXPLICIT_FILE = r"C:\Users\HARSHA\Documents\nstl\signal_generation\Project\Signal\PRI_11\channel001.dat"
#EXPLICIT_FILE = None

# ================================================================

cfg = Config()
set_seed(cfg.RANDOM_SEED)

checkpoint_path = _resolve_checkpoint(cfg, SOURCE)

# Use the correct signal length for each source
if SOURCE == "trial":
    # Override SIGNAL_LENGTH so run_inference loads the full trial recording
    cfg.SIGNAL_LENGTH = getattr(cfg, "TRIAL_SIGNAL_LENGTH", cfg.SIGNAL_LENGTH)

if EXPLICIT_FILE:
    input_path = EXPLICIT_FILE
elif SOURCE == "synthetic":
    input_path = _resolve_synthetic_file(cfg)
else:
    input_path = _resolve_trial_file(cfg, TRIAL_PRI, TRIAL_CHANNEL)

run_inference(cfg, SOURCE, checkpoint_path, input_path, cfg.INFERENCE_OUTPUT_DIR)

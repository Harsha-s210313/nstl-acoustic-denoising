"""
IDLE F5 Wrapper for Stage 2B Unsupervised Fine-Tuning.
Run directly by pressing F5 in Python IDLE without any command line arguments.
Requires Stage 2A pretrained checkpoint to exist.
"""

import sys
import os
import time

# Add package directory to sys.path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .config import Config
    from .denoiser_finetune import TrialDataFineTuner
except ImportError:
    from config import Config
    from denoiser_finetune import TrialDataFineTuner


def run() -> None:
    print("=" * 70)
    print("Launching Stage 2B Unsupervised Fine-Tuning (F5 IDLE Mode)...")
    print("Reminder: Ensure Stage 2A manual verification was completed successfully.")
    print("Waiting 3 seconds before commencing fine-tuning...")
    print("=" * 70)
    time.sleep(3)

    cfg = Config()
    finetuner = TrialDataFineTuner(cfg=cfg)
    finetuner.finetune()


if __name__ == "__main__":
    run()

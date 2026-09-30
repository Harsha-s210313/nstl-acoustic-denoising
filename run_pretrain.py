"""
IDLE F5 Wrapper for Stage 2A Supervised Pretraining.
Run directly by pressing F5 in Python IDLE without any command line arguments.
"""

import sys
import os

# Add package directory to sys.path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .config import Config
    from .denoiser_finetune import SyntheticDenoiserTrainer
except ImportError:
    from config import Config
    from denoiser_finetune import SyntheticDenoiserTrainer


def run() -> None:
    print("=" * 70)
    print("Launching Stage 2A Supervised U-Net Pretraining (F5 IDLE Mode)...")
    print("=" * 70)
    cfg = Config()
    trainer = SyntheticDenoiserTrainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    run()

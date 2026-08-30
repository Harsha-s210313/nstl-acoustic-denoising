"""
denoiser_pretrain.py
--------------------
Lightweight entry-point for Stage 2A: Synthetic U-Net Pretraining.

This file does NOT duplicate the training logic.
All training is delegated to SyntheticDenoiserTrainer in denoiser_finetune.py,
which is the CANONICAL implementation.

Usage:
    python denoiser_pretrain.py

Equivalent to:
    python denoiser_finetune.py --stage pretrain

After pretraining completes, MANUALLY VERIFY the pretrained model before
proceeding to Stage 2B (trial fine-tuning).
"""

import logging
import sys

from config import Config
from denoiser_finetune import SyntheticDenoiserTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


if __name__ == "__main__":
    log.info("=== Stage 2A: Synthetic U-Net Pretraining ===")
    log.info("(Canonical implementation: SyntheticDenoiserTrainer in denoiser_finetune.py)")
    cfg = Config()
    trainer = SyntheticDenoiserTrainer(cfg)
    trainer.run()

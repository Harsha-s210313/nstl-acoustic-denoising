# run_pretrain.py
# ---------------
# Open this file in IDLE and press F5 to run Stage 2A pretraining.
# No command-line arguments needed.

from config import Config
from denoiser_finetune import SyntheticDenoiserTrainer

cfg = Config()
trainer = SyntheticDenoiserTrainer(cfg)
trainer.run()

# run_finetune.py
# ---------------
# Open this file in IDLE and press F5 to run Stage 2B fine-tuning.
# No command-line arguments needed.
#
# ⚠ IMPORTANT: Only run this AFTER you have verified the pretrained model.
#   The pretrained checkpoint (models/unet_pretrained.pth) must exist.

from config import Config
from denoiser_finetune import TrialDataFineTuner

cfg = Config()
finetuner = TrialDataFineTuner(cfg)
finetuner.run()

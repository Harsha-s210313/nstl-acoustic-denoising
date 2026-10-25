"""
Headless End-to-End Test for Denoising Inference Pipeline.
Runs completely headless without requiring an interactive display or GUI window.
"""

import os
import sys
import tempfile
import numpy as np
import torch

try:
    from .config import Config
    from .utils import load_dat_file, save_dat_file, ensure_dir
    from .unet import Configurable1DUNet
    from .inference import run_pipeline
except (ImportError, ValueError):
    from config import Config
    from utils import load_dat_file, save_dat_file, ensure_dir
    from unet import Configurable1DUNet
    from inference import run_pipeline


def test_headless_inference_end_to_end():
    """
    Test headless execution of inference on both synthetic and trial sources.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = Config(
            SYNTHETIC_DATA_DIR=os.path.join(tmp_dir, "synthetic"),
            TRIAL_DATA_DIR=os.path.join(tmp_dir, "trial"),
            MODELS_DIR=os.path.join(tmp_dir, "models"),
            UNET_PRETRAINED_CHECKPOINT=os.path.join(tmp_dir, "models", "unet_pretrained_best.pth"),
            UNET_FINETUNED_CHECKPOINT=os.path.join(tmp_dir, "models", "unet_finetuned_best.pth"),
            INFERENCE_OUTPUT_DIR=os.path.join(tmp_dir, "inference_output"),
        )
        ensure_dir(cfg.MODELS_DIR)

        # 1. Create a dummy synthetic sample
        synth_clean_dir = os.path.join(cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_CLEAN_SUBDIR)
        synth_noisy_dir = os.path.join(cfg.SYNTHETIC_DATA_DIR, cfg.SYNTHETIC_NOISY_SUBDIR)
        ensure_dir(synth_clean_dir)
        ensure_dir(synth_noisy_dir)

        t = np.linspace(0, 0.25, cfg.SIGNAL_LENGTH, dtype=np.float32)
        clean_sig = np.sin(2 * np.pi * 20000 * t)
        noisy_sig = clean_sig + 0.2 * np.random.randn(len(t)).astype(np.float32)

        save_dat_file(os.path.join(synth_clean_dir, "synth_test.dat"), clean_sig)
        save_dat_file(os.path.join(synth_noisy_dir, "synth_test.dat"), noisy_sig)

        # 2. Create dummy checkpoint
        unet = Configurable1DUNet(cfg=cfg)
        torch.save({
            "tag": "pretrained",
            "model_state_dict": unet.state_dict(),
            "unet_config": cfg.UNET_CONFIG,
            "signal_length": cfg.SIGNAL_LENGTH,
        }, cfg.UNET_PRETRAINED_CHECKPOINT)

        # 3. Test Synthetic Headless Inference
        res_synth = run_pipeline(
            source="synthetic",
            model_type="pretrained",
            cfg=cfg,
            save_plot=True,
            device="cpu",
        )

        assert os.path.isfile(res_synth["output_dat"]), f"Missing output dat: {res_synth['output_dat']}"
        assert os.path.isfile(res_synth["output_plot"]), f"Missing plot: {res_synth['output_plot']}"
        denoised_data = load_dat_file(res_synth["output_dat"])
        assert len(denoised_data) == cfg.SIGNAL_LENGTH, f"Length mismatch: {len(denoised_data)} vs {cfg.SIGNAL_LENGTH}"
        assert os.path.getsize(res_synth["output_plot"]) > 1000

        # 4. Test Trial Headless Inference
        trial_sig_dir = os.path.join(cfg.TRIAL_DATA_DIR, cfg.TRIAL_SIGNAL_SUBDIR, cfg.INFERENCE_TRIAL_PRI)
        ensure_dir(trial_sig_dir)
        trial_raw = 0.1 * np.random.randn(19299).astype(np.float32)
        # Inject pulse
        trial_raw[6000:8011] += np.sin(2 * np.pi * 5000 * np.linspace(0, 0.113, 2011)).astype(np.float32)
        trial_file = os.path.join(trial_sig_dir, cfg.INFERENCE_TRIAL_CHANNEL)
        save_dat_file(trial_file, trial_raw)

        res_trial = run_pipeline(
            source="trial",
            model_type="pretrained",
            cfg=cfg,
            save_plot=True,
            device="cpu",
        )

        assert os.path.isfile(res_trial["output_dat"])
        assert os.path.isfile(res_trial["output_plot"])
        denoised_trial = load_dat_file(res_trial["output_dat"])
        assert len(denoised_trial) in (len(trial_raw), cfg.TRIAL_SIGNAL_LENGTH)
        assert os.path.getsize(res_trial["output_plot"]) > 1000

        print("\nAll headless inference tests passed successfully.")


if __name__ == "__main__":
    test_headless_inference_end_to_end()

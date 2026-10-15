"""
IDLE F5 Wrapper for Inference.
Run directly by pressing F5 in Python IDLE without any command line arguments.
Evaluates default dataset according to config.INFERENCE_DEFAULT_SOURCE.
"""

import sys
import os

# Add package directory to sys.path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .config import Config
    from .inference import run_pipeline
except ImportError:
    from config import Config
    from inference import run_pipeline


def run() -> None:
    print("=" * 70)
    print("Launching Acoustic Pipeline Inference (F5 IDLE Mode)...")
    print("=" * 70)
    cfg = Config()

    # Pretrained checkpoint provides robust supervised noise suppression
    # Finetuned checkpoint applies unsupervised noise reference adaptation
    model_choice = "finetuned"
    if not os.path.isfile(cfg.UNET_PRETRAINED_CHECKPOINT) and os.path.isfile(cfg.UNET_FINETUNED_CHECKPOINT):
        model_choice = "finetuned"

    res = run_pipeline(
        source=cfg.INFERENCE_DEFAULT_SOURCE,
        pri=cfg.INFERENCE_TRIAL_PRI,
        channel=cfg.INFERENCE_TRIAL_CHANNEL,
        model_type=model_choice,
        cfg=cfg,
        show_plot=True,
    )
    print("\nInference Complete:")
    print(f"  Input:            {res['input_path']}")
    print(f"  Model Stage:      {model_choice}")
    print(f"  Denoised Output:  {res['output_dat']}")
    print(f"  Diagnostic Plot:  {res['output_plot']}")
    print(f"  Detector:         {res['detector_verdict']}")
    print(f"  SNR Improvement:  {res['snr_improvement']:+.2f} dB")

    # Ensure plot opens on Windows
    if res.get("output_plot") and os.path.isfile(res["output_plot"]):
        if sys.platform.startswith("win"):
            try:
                os.startfile(res["output_plot"])
            except Exception:
                pass


if __name__ == "__main__":
    run()

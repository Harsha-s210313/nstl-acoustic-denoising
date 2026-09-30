"""
Thin entry point for Stage 2A Supervised U-Net Pretraining.
Calls SyntheticDenoiserTrainer from denoiser_finetune.py.
"""

import argparse
try:
    from .config import Config
    from .denoiser_finetune import SyntheticDenoiserTrainer
except (ImportError, ValueError):
    from config import Config
    from denoiser_finetune import SyntheticDenoiserTrainer


def main():
    parser = argparse.ArgumentParser(description="Stage 2A U-Net Pretrainer")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    args = parser.parse_args()

    cfg = Config()
    trainer = SyntheticDenoiserTrainer(cfg=cfg, device=args.device)
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

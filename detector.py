"""
detector.py
-----------
Stage 1: SignalDetectorCNN
A lightweight 1D CNN binary classifier that determines whether an acoustic
pulse/echo is present in a fixed-length signal window.

Architecture overview
---------------------
  [Conv1d → BN → ReLU → MaxPool] × N_layers
  → Global Average Pooling
  → FC hidden → ReLU → Dropout → FC output (2 logits)

All architectural parameters come from Config:
  - DETECTOR_FEATURES   : list of channel widths per conv block
  - DETECTOR_KERNEL_SIZE: convolution kernel size
  - DETECTOR_DROPOUT    : dropout probability before the output layer
  - DETECTOR_FC_HIDDEN  : width of the fully-connected hidden layer

The model is Jetson-friendly: no exotic ops, pure PyTorch.
"""

import torch
import torch.nn as nn

from config import Config


class _ConvBlock(nn.Module):
    """Conv1d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 4):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SignalDetectorCNN(nn.Module):
    """
    1D CNN binary classifier for acoustic signal detection.

    Input  : (batch, 1, SIGNAL_LENGTH) — one-channel real-valued signal
    Output : (batch, 2)                — raw logits [P(noise), P(signal)]

    Use torch.argmax(logits, dim=1) or torch.sigmoid(logits[:, 1]) > 0.5
    for binary predictions.

    Parameters
    ----------
    cfg : Config
        Configuration object.  Architecture is read from:
          cfg.DETECTOR_FEATURES, cfg.DETECTOR_KERNEL_SIZE,
          cfg.DETECTOR_FC_HIDDEN, cfg.DETECTOR_DROPOUT.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        features = cfg.DETECTOR_FEATURES
        kernel = cfg.DETECTOR_KERNEL_SIZE
        fc_hidden = cfg.DETECTOR_FC_HIDDEN
        dropout_p = cfg.DETECTOR_DROPOUT

        # Build convolutional blocks
        layers = []
        in_ch = 1
        for out_ch in features:
            layers.append(_ConvBlock(in_ch, out_ch, kernel))
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # Global average pooling collapses the time dimension → (batch, C)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_hidden, 2),          # 2 output logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, L) — single-channel 1D signal.

        Returns
        -------
        torch.Tensor
            Shape (B, 2) — raw logits.
        """
        feats = self.encoder(x)   # (B, C_last, L')
        pooled = self.gap(feats)  # (B, C_last, 1)
        logits = self.classifier(pooled)  # (B, 2)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions (0 or 1) without computing gradients."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

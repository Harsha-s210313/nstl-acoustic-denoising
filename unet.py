"""
unet.py
-------
Stage 2: Configurable1DUNet
A configurable 1D U-Net for acoustic signal denoising.

Architecture
------------
  Encoder: N conv-blocks (each = Conv → BN → ReLU → Conv → BN → ReLU),
            followed by MaxPool downsampling.
  Bottleneck: one conv-block at the coarsest scale.
  Decoder: N upsample + skip-concat + conv-block stages.
  Output: 1×1 Conv to map back to ``out_channels``.

The number of levels and channel widths are fully determined by
``Config.UNET_CONFIG["features"]``.  E.g.:

    features = [16, 32, 64, 128]  →  4 encoder levels + bottleneck

No code changes are needed to switch from [16,32,64,128] to [32,64,128,256].

Input/output shapes
-------------------
  Input  : (B, in_channels,  SIGNAL_LENGTH)
  Output : (B, out_channels, SIGNAL_LENGTH)

The output length always matches the input length: the decoder uses
``F.interpolate`` (linear mode) before each skip-connection merge to handle
any odd-length artefacts from integer downsampling.

Dropout is applied in each conv-block (after the second convolution) with
probability ``Config.UNET_CONFIG["dropout"]``.  Set to 0.0 to disable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from config import Config


# --------------------------------------------------------------------------- #
# Building-block
# --------------------------------------------------------------------------- #

class _DoubleConv(nn.Module):
    """
    Two sequential Conv1d → BatchNorm1d → ReLU blocks, with optional Dropout
    after the second convolution.

    Parameters
    ----------
    in_ch, out_ch : int
        Input and output channel counts.
    kernel : int
        Convolution kernel size (must be odd for symmetric padding).
    dropout : float
        Dropout probability (0.0 = disabled).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #

class Configurable1DUNet(nn.Module):
    """
    Fully configurable 1D U-Net for signal denoising.

    All architecture choices come from Config.UNET_CONFIG:
        features    : list of encoder channel widths (one entry = one level)
        kernel_size : convolution kernel (odd integer)
        dropout     : dropout probability
        in_channels : input channels (1 for real-valued signals)
        out_channels: output channels (1 for real-valued signals)

    Input  : (B, in_channels,  L)
    Output : (B, out_channels, L)   — same length L as input

    Parameters
    ----------
    cfg : Config
        Configuration object.

    Raises
    ------
    ValueError
        If ``features`` is empty or ``kernel_size`` is even.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        unet_cfg = cfg.UNET_CONFIG
        features: List[int] = unet_cfg["features"]
        kernel: int = unet_cfg["kernel_size"]
        dropout: float = unet_cfg["dropout"]
        in_ch: int = unet_cfg["in_channels"]
        out_ch: int = unet_cfg["out_channels"]

        if not features:
            raise ValueError("UNET_CONFIG['features'] must be a non-empty list.")
        if kernel % 2 == 0:
            raise ValueError(
                f"UNET_CONFIG['kernel_size'] must be odd, got {kernel}."
            )

        self.features = features
        self.n_levels = len(features)

        # ------------------------------------------------------------------ #
        # Encoder
        # ------------------------------------------------------------------ #
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        prev_ch = in_ch
        for feat in features:
            self.encoder_blocks.append(_DoubleConv(prev_ch, feat, kernel, dropout))
            prev_ch = feat

        # ------------------------------------------------------------------ #
        # Bottleneck (one level deeper, double the last feature count)
        # ------------------------------------------------------------------ #
        bottleneck_ch = features[-1] * 2
        self.bottleneck = _DoubleConv(features[-1], bottleneck_ch, kernel, dropout)

        # ------------------------------------------------------------------ #
        # Decoder
        # ------------------------------------------------------------------ #
        # At each decoder level:
        #   in_ch = bottleneck_ch (or prev decoder_ch) + skip_ch
        # The upsampled tensor is concatenated with the matching encoder skip.
        self.decoder_upconvs = nn.ModuleList()   # 1×1 projection before merge
        self.decoder_blocks = nn.ModuleList()

        up_ch = bottleneck_ch
        for feat in reversed(features):
            # Projection: halve the channels of the upsampled tensor
            self.decoder_upconvs.append(
                nn.Conv1d(up_ch, feat, kernel_size=1, bias=False)
            )
            # After concatenation with skip: feat + feat = 2*feat input channels
            self.decoder_blocks.append(_DoubleConv(2 * feat, feat, kernel, dropout))
            up_ch = feat

        # ------------------------------------------------------------------ #
        # Output head
        # ------------------------------------------------------------------ #
        self.out_conv = nn.Conv1d(features[0], out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, in_channels, L).

        Returns
        -------
        torch.Tensor
            Shape (B, out_channels, L)  — same length as input.
        """
        # Store input length for final interpolation guard
        input_len = x.shape[-1]

        # ------------------------------------------------------------------ #
        # Encoder pass — collect skip connections
        # ------------------------------------------------------------------ #
        skips: List[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # ------------------------------------------------------------------ #
        # Bottleneck
        # ------------------------------------------------------------------ #
        x = self.bottleneck(x)

        # ------------------------------------------------------------------ #
        # Decoder pass — reverse skip order
        # ------------------------------------------------------------------ #
        for i, (up_conv, dec_block) in enumerate(
            zip(self.decoder_upconvs, self.decoder_blocks)
        ):
            skip = skips[self.n_levels - 1 - i]

            # Upsample to match skip length (handles odd-length artefacts)
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = up_conv(x)                         # project channels

            x = torch.cat([skip, x], dim=1)        # channel-wise concat
            x = dec_block(x)

        # ------------------------------------------------------------------ #
        # Output head — ensure final length == input length
        # ------------------------------------------------------------------ #
        x = self.out_conv(x)
        if x.shape[-1] != input_len:
            x = F.interpolate(x, size=input_len, mode="linear", align_corners=False)

        return x

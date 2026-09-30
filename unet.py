"""
Configurable 1-D U-Net for Underwater Acoustic Signal Denoising.
Fully convolutional architecture supporting arbitrary input lengths
(e.g., L=25000, L=5000, and odd lengths such as L=4999).
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import Config, default_config
except (ImportError, ValueError):
    from config import Config, default_config


class _DoubleConv(nn.Module):
    """
    (Conv1d(k, pad=k//2, bias=False) -> BN -> ReLU) x 2,
    optionally followed by Dropout if dropout > 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Configurable1DUNet(nn.Module):
    """
    Fully-convolutional 1D U-Net driven entirely by UNET_CONFIG.
    Maintains skip connections, linear interpolation upsampling with 1x1 projection,
    and a final length-preservation guard.
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        unet_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        c = unet_config if unet_config is not None else (cfg.UNET_CONFIG if cfg else default_config.UNET_CONFIG)

        features: List[int] = c.get("features", [16, 32, 64, 128])
        kernel_size: int = c.get("kernel_size", 7)
        dropout: float = float(c.get("dropout", 0.1))
        in_channels: int = c.get("in_channels", 1)
        out_channels: int = c.get("out_channels", 1)

        # Validation
        if not features or not all(isinstance(f, int) and f > 0 for f in features):
            raise ValueError(f"UNet features must be a non-empty list of positive ints, got {features}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"UNet kernel_size must be a positive odd integer, got {kernel_size}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"UNet dropout must be in [0, 1), got {dropout}")

        self.features = features
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder path
        self.encoders = nn.ModuleList()
        curr_in = in_channels
        for feat in features:
            self.encoders.append(_DoubleConv(curr_in, feat, kernel_size, dropout))
            curr_in = feat

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = _DoubleConv(
            features[-1],
            features[-1] * 2,
            kernel_size,
            dropout,
        )

        # Decoder path
        self.up_projections = nn.ModuleList()
        self.decoders = nn.ModuleList()

        prev_feat = features[-1] * 2
        for feat in reversed(features):
            # 1x1 conv projection to reduce channel dimensions after interpolation
            self.up_projections.append(nn.Conv1d(prev_feat, feat, kernel_size=1, bias=False))
            # DoubleConv takes concatenated channels (skip `feat` + projected `feat` = 2*feat) down to `feat`
            self.decoders.append(_DoubleConv(feat * 2, feat, kernel_size, dropout))
            prev_feat = feat

        # Output projection
        self.out_conv = nn.Conv1d(features[0], out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects x: (B, in_channels, L) or (B, L)
        Returns: (B, out_channels, L)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        orig_len = x.shape[-1]
        skips: List[torch.Tensor] = []

        # Encoder forward
        out = x
        for enc in self.encoders:
            out = enc(out)
            skips.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder forward
        for up_proj, dec, skip in zip(self.up_projections, self.decoders, reversed(skips)):
            # Linear interpolate to match exact skip length
            out = F.interpolate(out, size=skip.shape[-1], mode="linear", align_corners=False)
            out = up_proj(out)
            out = torch.cat([skip, out], dim=1)
            out = dec(out)

        # 1x1 output conv
        out = self.out_conv(out)

        # Final guard so output length strictly matches input length
        if out.shape[-1] != orig_len:
            out = F.interpolate(out, size=orig_len, mode="linear", align_corners=False)

        return out

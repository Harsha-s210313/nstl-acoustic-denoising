"""
Stage 1: 1-D CNN Detector for Acoustic Chirp Pulses.
Identifies presence of active LFM pulses buried in ambient acoustic noise.
"""

from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
try:
    from .config import Config, default_config
except (ImportError, ValueError):
    from config import Config, default_config


class SignalDetectorCNN(nn.Module):
    """
    Lightweight 1D CNN Detector.
    Stack of [Conv1d(k=7, pad=3) -> BN -> ReLU -> MaxPool1d(4)]
    followed by AdaptiveAvgPool1d(1), dense layer, dropout, and 2-class logits.
    """

    def __init__(
        self,
        cfg: Optional[Config] = None,
        features: Optional[List[int]] = None,
        kernel_size: Optional[int] = None,
        dropout: Optional[float] = None,
        fc_hidden: Optional[int] = None,
        in_channels: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        config = cfg or default_config

        feat_list = features if features is not None else config.DETECTOR_FEATURES
        k = kernel_size if kernel_size is not None else config.DETECTOR_KERNEL_SIZE
        drop_p = dropout if dropout is not None else config.DETECTOR_DROPOUT
        fc_dim = fc_hidden if fc_hidden is not None else config.DETECTOR_FC_HIDDEN

        if not feat_list or not all(isinstance(f, int) and f > 0 for f in feat_list):
            raise ValueError("Detector features must be a non-empty list of positive integers")
        if k <= 0 or k % 2 == 0:
            raise ValueError(f"Detector kernel_size must be a positive odd integer, got {k}")

        layers = []
        curr_in = in_channels
        for out_ch in feat_list:
            layers.append(
                nn.Conv1d(
                    curr_in,
                    out_ch,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=4, stride=4))
            curr_in = out_ch

        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_list[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_p),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects x: (B, 1, L) or (B, L)
        Returns logits: (B, 2)
        """
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        feats = self.conv_blocks(x)
        pooled = self.global_pool(feats)
        logits = self.classifier(pooled)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x: Union[torch.Tensor, "np.ndarray"],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference helper under torch.no_grad().
        Returns:
            preds: (B,) class predictions (0=Noise, 1=Signal)
            probs: (B, 2) softmax probabilities
        """
        self.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        device = next(self.parameters()).device
        x = x.to(device)

        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs

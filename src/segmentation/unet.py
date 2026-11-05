"""Compact U-Net for fish presence segmentation (CPU-friendly, ~1.9 M params).

Design notes
------------
- Encoder channels 16/32/64/128, bottleneck 256 — small enough to train on a
  laptop CPU and fast enough for near-real-time video inference.
- BatchNorm + ReLU double-conv blocks, transposed-conv upsampling with skip
  concatenation (Ronneberger et al., 2015).
- Single-logit output; apply sigmoid externally (keeps BCEWithLogits fusion).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    """U-Net with 4 downsampling stages and a single-channel logit output."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 16) -> None:
        super().__init__()
        b1, b2, b3, b4 = base, base * 2, base * 4, base * 8
        self.enc1 = DoubleConv(in_channels, b1)
        self.enc2 = DoubleConv(b1, b2)
        self.enc3 = DoubleConv(b2, b3)
        self.enc4 = DoubleConv(b3, b4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b4, b4 * 2)

        self.up4 = nn.ConvTranspose2d(b4 * 2, b4, 2, stride=2)
        self.dec4 = DoubleConv(b4 * 2, b4)
        self.up3 = nn.ConvTranspose2d(b4, b3, 2, stride=2)
        self.dec3 = DoubleConv(b3 * 2, b3)
        self.up2 = nn.ConvTranspose2d(b3, b2, 2, stride=2)
        self.dec2 = DoubleConv(b2 * 2, b2)
        self.up1 = nn.ConvTranspose2d(b2, b1, 2, stride=2)
        self.dec1 = DoubleConv(b1 * 2, b1)
        self.head = nn.Conv2d(b1, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

    @torch.no_grad()
    def predict_mask(self, image_rgb_uint8: "np.ndarray", input_size: int = 256) -> "np.ndarray":
        """Probability mask [H, W] in [0, 1] for an RGB uint8 image."""
        import cv2
        import numpy as np

        was_training = self.training
        self.eval()
        orig_h, orig_w = image_rgb_uint8.shape[:2]
        img = cv2.resize(image_rgb_uint8, (input_size, input_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]
        prob = torch.sigmoid(self(x))[0, 0].cpu().numpy()
        if was_training:
            self.train()
        return cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

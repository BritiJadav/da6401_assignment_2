"""VGG11 encoder."""

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    Follows the official VGG-11 architecture (Simonyan & Zisserman,
    https://arxiv.org/abs/1409.1556). BatchNorm is injected after every
    Conv layer (design choice). Input size is fixed at 224x224 per the paper.

    Spatial sizes for 224x224 input:
        block1 : [B,  64, 224, 224]  (before pool1)
        block2 : [B, 128, 112, 112]  (before pool2)
        block3 : [B, 256,  56,  56]  (before pool3)
        block4 : [B, 512,  28,  28]  (before pool4)
        block5 : [B, 512,  14,  14]  (before pool5)
        bottleneck : [B, 512, 7, 7]  (after  pool5)
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # Block 1 — 1 conv, 64 filters
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2 — 1 conv, 128 filters
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3 — 2 convs, 256 filters
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4 — 2 convs, 512 filters
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5 — 2 convs, 512 filters
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x:               Input image tensor [B, 3, H, W].
            return_features: If True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B, 512, 7, 7].
            - if return_features=True:  (bottleneck, feature_dict)
                feature_dict keys: "block1" … "block5"
        """
        features: Dict[str, torch.Tensor] = {}

        x = self.block1(x);  features["block1"] = x;  x = self.pool1(x)
        x = self.block2(x);  features["block2"] = x;  x = self.pool2(x)
        x = self.block3(x);  features["block3"] = x;  x = self.pool3(x)
        x = self.block4(x);  features["block4"] = x;  x = self.pool4(x)
        x = self.block5(x);  features["block5"] = x;  x = self.pool5(x)

        if return_features:
            return x, features
        return x


# Alias — autograder uses: from models.vgg11 import VGG11
VGG11 = VGG11Encoder

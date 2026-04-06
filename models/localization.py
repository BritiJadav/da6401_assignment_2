import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Encoder
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 4)  # x, y, w, h
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model."""

        # Extract features
        x = self.encoder(x)   # [B, 512, 7, 7]

        # Flatten
        x = torch.flatten(x, 1)

        # Predict bounding box
        x = self.regressor(x)

        return x
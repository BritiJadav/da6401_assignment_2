import torch
import torch.nn as nn

from models.vgg11 import VGG11

# Input size fixed per VGG paper (224x224)
_FLAT_DIM = 512 * 7 * 7


class VGG11Localizer(nn.Module):
    """VGG11 encoder + bounding-box regression head.

    Output: [x_center, y_center, width, height] in pixel space (not normalized).
    No Sigmoid — raw regression output lets the network predict pixel coordinates.

    Args:
        in_channels: Number of input image channels (default: 3).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)

        # Regression head — outputs pixel-space (cx, cy, w, h)
        self.regressor = nn.Sequential(
            nn.Linear(_FLAT_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),      # (x_center, y_center, width, height)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            boxes: [B, 4]  (x_center, y_center, width, height) in pixel space
        """
        x = self.encoder(x)          # [B, 512, 7, 7]
        x = torch.flatten(x, 1)      # [B, 512*7*7]
        x = self.regressor(x)        # [B, 4]
        return x

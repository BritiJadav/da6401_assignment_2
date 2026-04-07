import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg11 import VGG11


class DecoderBlock(nn.Module):
    """Transposed-conv upsampling + skip connection + double-conv block.

    Handles spatial size mismatches via center-crop (upsampled too big)
    or zero-pad (upsampled too small). No bilinear interpolation.

    Args:
        in_channels:   Channels of the upsampled feature map.
        skip_channels: Channels of the skip-connection feature map.
        out_channels:  Channels produced by this block.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Align spatial dimensions
        if x.shape[2] > skip.shape[2] or x.shape[3] > skip.shape[3]:
            dh = x.shape[2] - skip.shape[2]
            dw = x.shape[3] - skip.shape[3]
            x = x[
                :, :,
                dh // 2 : x.shape[2] - (dh - dh // 2),
                dw // 2 : x.shape[3] - (dw - dw // 2),
            ]
        elif x.shape[2] < skip.shape[2] or x.shape[3] < skip.shape[3]:
            pad_h = skip.shape[2] - x.shape[2]
            pad_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation network with VGG11 encoder.

    Encoder spatial sizes for 224x224 input:
        block1: [B,  64, 224, 224]   (before pool1)
        block2: [B, 128, 112, 112]   (before pool2)
        block3: [B, 256,  56,  56]   (before pool3)
        block4: [B, 512,  28,  28]   (before pool4)
        block5: [B, 512,  14,  14]   (before pool5)
        bottleneck: [B, 512, 7, 7]   (after  pool5)

    Decoder path:
        dec4: 7   → 14,  skip=block5
        dec3: 14  → 28,  skip=block4
        dec2: 28  → 56,  skip=block3
        dec1: 56  → 112, skip=block2
        dec0: 112 → 224, skip=block1

    Args:
        num_classes: Number of segmentation classes (default: 3).
        in_channels: Number of input image channels (default: 3).
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()

        self.encoder = VGG11(in_channels=in_channels)

        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 128, 64)
        self.dec0 = DecoderBlock(64,   64,  64)

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            seg_logits: [B, num_classes, 224, 224]
        """
        bottleneck, features = self.encoder(x, return_features=True)

        f1 = features["block1"]
        f2 = features["block2"]
        f3 = features["block3"]
        f4 = features["block4"]
        f5 = features["block5"]

        x = self.dec4(bottleneck, f5)
        x = self.dec3(x, f4)
        x = self.dec2(x, f3)
        x = self.dec1(x, f2)
        x = self.dec0(x, f1)

        return self.seg_head(x)

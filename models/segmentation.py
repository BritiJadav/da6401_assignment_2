import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg11 import VGG11Encoder


class DecoderBlock(nn.Module):
    """Transposed-conv upsampling + skip connection + conv block.

    Handles size mismatches via center-crop (x too big) or
    zero-padding (x too small). No bilinear interpolation used.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2] > skip.shape[2] or x.shape[3] > skip.shape[3]:
            # x is too big → center-crop to match skip
            dh = x.shape[2] - skip.shape[2]
            dw = x.shape[3] - skip.shape[3]
            x = x[
                :, :,
                dh // 2 : x.shape[2] - (dh - dh // 2),
                dw // 2 : x.shape[3] - (dw - dw // 2),
            ]
        elif x.shape[2] < skip.shape[2] or x.shape[3] < skip.shape[3]:
            # x is too small → zero-pad to match skip
            pad_h = skip.shape[2] - x.shape[2]
            pad_w = skip.shape[3] - x.shape[3]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation network with VGG11 encoder.

    Encoder spatial sizes for 224x224 input:
        block1: [B,  64, 224, 224]  (before pool1)
        block2: [B, 128, 112, 112]  (before pool2)
        block3: [B, 256,  56,  56]  (before pool3)
        block4: [B, 512,  28,  28]  (before pool4)
        block5: [B, 512,  14,  14]  (before pool5)
        bottleneck: [B, 512, 7, 7]  (after pool5)

    Decoder path:
        dec4: 7   → 14,  skip=block5 [B, 512, 14, 14]
        dec3: 14  → 28,  skip=block4 [B, 512, 28, 28]
        dec2: 28  → 56,  skip=block3 [B, 256, 56, 56]
        dec1: 56  → 112, skip=block2 [B, 128, 112, 112]
        dec0: 112 → 224, skip=block1 [B,  64, 224, 224]
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.dec4 = DecoderBlock(512, 512, 512)  # bottleneck + block5
        self.dec3 = DecoderBlock(512, 512, 256)  # dec4 out   + block4
        self.dec2 = DecoderBlock(256, 256, 128)  # dec3 out   + block3
        self.dec1 = DecoderBlock(128, 128, 64)   # dec2 out   + block2
        self.dec0 = DecoderBlock(64,  64,  64)   # dec1 out   + block1

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, features = self.encoder(x, return_features=True)

        f1 = features["block1"]  # [B,  64, 224, 224]
        f2 = features["block2"]  # [B, 128, 112, 112]
        f3 = features["block3"]  # [B, 256,  56,  56]
        f4 = features["block4"]  # [B, 512,  28,  28]
        f5 = features["block5"]  # [B, 512,  14,  14]

        x = self.dec4(bottleneck, f5)  # 7   → 14
        x = self.dec3(x, f4)           # 14  → 28
        x = self.dec2(x, f3)           # 28  → 56
        x = self.dec1(x, f2)           # 56  → 112
        x = self.dec0(x, f1)           # 112 → 224

        x = self.seg_head(x)           # [B, num_classes, 224, 224]
        return x

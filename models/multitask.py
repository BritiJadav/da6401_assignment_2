import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import DecoderBlock


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    A single VGG11 encoder runs ONCE per forward pass.
    All three heads share the same backbone features.
    """

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super().__init__()

        # Shared encoder — runs only once per forward pass
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(0.5),

            nn.Linear(4096, num_breeds)
        )

        # Localization head
        self.localizer = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4)
            nn.Sigmoid()
        )

        # Segmentation decoder only (reuses shared encoder features)
        # Channel sizes match actual VGG11 encoder outputs:
        #   bottleneck: 512, block5: 512, block4: 512,
        #   block3: 256, block2: 128, block1: 64
        self.dec4 = DecoderBlock(512, 512, 512)  # bottleneck + block5
        self.dec3 = DecoderBlock(512, 512, 256)  # dec4 out   + block4
        self.dec2 = DecoderBlock(256, 256, 128)  # dec3 out   + block3
        self.dec1 = DecoderBlock(128, 128, 64)   # dec2 out   + block2
        self.dec0 = DecoderBlock(64,  64,  64)   # dec1 out   + block1

        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """Single forward pass — encoder runs exactly once.

        Returns:
            dict with keys: 'classification', 'localization', 'segmentation'
        """

        # ONE encoder pass — all features shared
        bottleneck, features = self.encoder(x, return_features=True)

        f1 = features["block1"]  # [B,  64, 224, 224]
        f2 = features["block2"]  # [B, 128, 112, 112]
        f3 = features["block3"]  # [B, 256,  56,  56]
        f4 = features["block4"]  # [B, 512,  28,  28]
        f5 = features["block5"]  # [B, 512,  14,  14]

        # Flatten for FC heads
        flat = torch.flatten(bottleneck, 1)

        # Classification
        cls_out = self.classifier(flat)

        # Localization
        loc_out = self.localizer(flat)

        # Segmentation decoder
        d = self.dec4(bottleneck, f5)  # 7   → 14
        d = self.dec3(d, f4)           # 14  → 28
        d = self.dec2(d, f3)           # 28  → 56
        d = self.dec1(d, f2)           # 56  → 112
        d = self.dec0(d, f1)           # 112 → 224
        seg_out = self.seg_head(d)     # [B, seg_classes, 224, 224]

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }

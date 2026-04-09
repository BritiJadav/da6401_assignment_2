import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown

from models.vgg11 import VGG11

_CKPT_DIR      = "checkpoints"
_CKPT_UNET     = os.path.join(_CKPT_DIR, "unet.pth")
_DRIVE_ID_UNET = "1Rqkyv08xU8DyzAXke70uDFvXGxXnOEkS"


class DecoderBlock(nn.Module):
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
    def __init__(self, num_classes: int = 3, in_channels: int = 3,
                 load_ckpt: bool = True):
        super().__init__()

        self.encoder  = VGG11(in_channels=in_channels)
        self.dec4     = DecoderBlock(512, 512, 512)
        self.dec3     = DecoderBlock(512, 512, 256)
        self.dec2     = DecoderBlock(256, 256, 128)
        self.dec1     = DecoderBlock(128, 128,  64)
        self.dec0     = DecoderBlock( 64,  64,  64)
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

        if load_ckpt:
            self._load_checkpoint()

    def _load_checkpoint(self):
        os.makedirs(_CKPT_DIR, exist_ok=True)
        if not os.path.exists(_CKPT_UNET):
            print("[UNet] Downloading unet.pth ...")
            gdown.download(id=_DRIVE_ID_UNET,
                           output=_CKPT_UNET, quiet=False)
        if os.path.exists(_CKPT_UNET):
            ckpt = torch.load(_CKPT_UNET,
                              map_location=torch.device("cpu"),
                              weights_only=False)
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.load_state_dict(sd)
            print(f"[UNet] ✅ Loaded checkpoint from {_CKPT_UNET}")
        else:
            print(f"[UNet] ⚠️ {_CKPT_UNET} not found.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

import os
import torch
import torch.nn as nn
import gdown

from models.vgg11 import VGG11

_FLAT_DIM            = 512 * 7 * 7
_CKPT_DIR            = "checkpoints"
_CKPT_LOCALIZER      = os.path.join(_CKPT_DIR, "localizer.pth")
_DRIVE_ID_LOCALIZER  = "1mVFkwL0KEW-Eo4gd5Y7izVMR-ZYHa0_D"


class VGG11Localizer(nn.Module):
    def __init__(self, in_channels: int = 3, load_ckpt: bool = True):
        super().__init__()

        self.encoder    = VGG11(in_channels=in_channels)
        self.image_size = 224

        # Matches checkpoint exactly: 25088 → 512 → 4 + Sigmoid
        self.regressor = nn.Sequential(
            nn.Linear(_FLAT_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

        if load_ckpt:
            self._load_checkpoint()

    def _load_checkpoint(self):
        os.makedirs(_CKPT_DIR, exist_ok=True)
        if not os.path.exists(_CKPT_LOCALIZER):
            print("[Localizer] Downloading localizer.pth ...")
            gdown.download(id=_DRIVE_ID_LOCALIZER,
                           output=_CKPT_LOCALIZER, quiet=False)
        if os.path.exists(_CKPT_LOCALIZER):
            ckpt = torch.load(_CKPT_LOCALIZER,
                              map_location=torch.device("cpu"),
                              weights_only=False)
            sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.load_state_dict(sd)
            print(f"[Localizer] ✅ Loaded checkpoint from {_CKPT_LOCALIZER}")
        else:
            print(f"[Localizer] ⚠️ {_CKPT_LOCALIZER} not found.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.regressor(x) * self.image_size

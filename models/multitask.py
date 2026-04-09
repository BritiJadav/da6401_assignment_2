import os
import gdown
import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import DecoderBlock

_FLAT_DIM = 512 * 7 * 7

_CKPT_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
_CKPT_CLASSIFIER = os.path.join(_CKPT_DIR, "classifier.pth")
_CKPT_LOCALIZER  = os.path.join(_CKPT_DIR, "localizer.pth")
_CKPT_UNET       = os.path.join(_CKPT_DIR, "unet.pth")

_DRIVE_ID_CLASSIFIER = "177Z83QbYUteiS6bgW6FdmQfCiYTGesvu"
_DRIVE_ID_LOCALIZER  = "1mVFkwL0KEW-Eo4gd5Y7izVMR-ZYHa0_D"
_DRIVE_ID_UNET       = "1Rqkyv08xU8DyzAXke70uDFvXGxXnOEkS"


def _download_checkpoints():
    os.makedirs(_CKPT_DIR, exist_ok=True)
    if not os.path.exists(_CKPT_CLASSIFIER):
        print("[MultiTask] Downloading classifier.pth ...")
        gdown.download(id=_DRIVE_ID_CLASSIFIER, output=_CKPT_CLASSIFIER, quiet=False)
    if not os.path.exists(_CKPT_LOCALIZER):
        print("[MultiTask] Downloading localizer.pth ...")
        gdown.download(id=_DRIVE_ID_LOCALIZER, output=_CKPT_LOCALIZER, quiet=False)
    if not os.path.exists(_CKPT_UNET):
        print("[MultiTask] Downloading unet.pth ...")
        gdown.download(id=_DRIVE_ID_UNET, output=_CKPT_UNET, quiet=False)


def _load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        load_ckpts: bool = True,
    ):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.classifier = nn.Sequential(
            nn.Linear(_FLAT_DIM, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_breeds),
        )

        self.localizer = nn.Sequential(
            nn.Linear(_FLAT_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 128,  64)
        self.dec0 = DecoderBlock( 64,  64,  64)
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        if load_ckpts:
            _download_checkpoints()
            self._load_checkpoints()

    def _load_checkpoints(self):
        # Load classifier checkpoint — also use its encoder
        if os.path.exists(_CKPT_CLASSIFIER):
            sd = _load_state_dict(_CKPT_CLASSIFIER)

            # Load encoder from classifier checkpoint ONLY (no averaging)
            enc_sd = {k[len("encoder."):]: v
                      for k, v in sd.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc_sd)
            print(f"[MultiTask] Shared encoder loaded from classifier checkpoint.")

            cls_sd = {k[len("classifier."):]: v
                      for k, v in sd.items() if k.startswith("classifier.")}
            self.classifier.load_state_dict(cls_sd)
            print(f"[MultiTask] Loaded classifier head from {_CKPT_CLASSIFIER}")
        else:
            print(f"[MultiTask] WARNING: {_CKPT_CLASSIFIER} not found.")

        # Load localizer head only (no encoder)
        if os.path.exists(_CKPT_LOCALIZER):
            sd = _load_state_dict(_CKPT_LOCALIZER)
            loc_sd = {k[len("regressor."):]: v
                      for k, v in sd.items() if k.startswith("regressor.")}
            self.localizer.load_state_dict(loc_sd)
            print(f"[MultiTask] Loaded localizer head from {_CKPT_LOCALIZER}")
        else:
            print(f"[MultiTask] WARNING: {_CKPT_LOCALIZER} not found.")

        # Load UNet decoder only (no encoder)
        if os.path.exists(_CKPT_UNET):
            sd = _load_state_dict(_CKPT_UNET)
            for prefix in ("dec4", "dec3", "dec2", "dec1", "dec0", "seg_head"):
                block_sd = {k[len(prefix) + 1:]: v
                            for k, v in sd.items() if k.startswith(prefix + ".")}
                getattr(self, prefix).load_state_dict(block_sd)
            print(f"[MultiTask] Loaded UNet decoder from {_CKPT_UNET}")
        else:
            print(f"[MultiTask] WARNING: {_CKPT_UNET} not found.")

    def forward(self, x: torch.Tensor) -> dict:
        bottleneck, features = self.encoder(x, return_features=True)

        f1 = features["block1"]
        f2 = features["block2"]
        f3 = features["block3"]
        f4 = features["block4"]
        f5 = features["block5"]

        flat = torch.flatten(bottleneck, 1)

        cls_out = self.classifier(flat)
        loc_out = self.localizer(flat) * 224.0

        d = self.dec4(bottleneck, f5)
        d = self.dec3(d, f4)
        d = self.dec2(d, f3)
        d = self.dec1(d, f2)
        d = self.dec0(d, f1)
        seg_out = self.seg_head(d)

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }

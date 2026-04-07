import os
import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import DecoderBlock

# Input size fixed per VGG paper
_FLAT_DIM = 512 * 7 * 7

# Relative paths to the three task checkpoints
_CKPT_DIR        = "checkpoints"
_CKPT_CLASSIFIER = os.path.join(_CKPT_DIR, "classifier.pth")
_CKPT_LOCALIZER  = os.path.join(_CKPT_DIR, "localizer.pth")
_CKPT_UNET       = os.path.join(_CKPT_DIR, "unet.pth")


def _load_state_dict(path: str, device: torch.device) -> dict:
    """Load a checkpoint that is either a plain state_dict or a dict
    containing a 'state_dict' key (as saved by train_checkpoints.py)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Architecture
    ------------
    • One VGG11 encoder (shared, runs exactly once per forward pass).
    • Classification head  → breed logits  [B, num_breeds]
    • Localization head    → bbox          [B, 4]  pixel-space (cx, cy, w, h)
    • Segmentation decoder → pixel logits  [B, seg_classes, 224, 224]

    Args:
        num_breeds:  Number of breed classes (default: 37).
        seg_classes: Number of segmentation classes (default: 3).
        in_channels: Input image channels (default: 3).
        dropout_p:   Dropout probability for classification head (default: 0.5).
        load_ckpts:  Whether to load task checkpoints (default: True).
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        load_ckpts: bool = True,
    ):
        super().__init__()

        # ── Shared encoder ───────────────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(_FLAT_DIM, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),

            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ────────────────────────────────────────────────
        # Output: pixel-space (cx, cy, w, h) — no Sigmoid
        self.localizer = nn.Sequential(
            nn.Linear(_FLAT_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

        # ── Segmentation decoder ─────────────────────────────────────────────
        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 128,  64)
        self.dec0 = DecoderBlock( 64,  64,  64)
        self.seg_head = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ── Load checkpoint weights (only if flag is set) ────────────────────
        if load_ckpts:
            self._load_checkpoints()

    # ── Checkpoint loading ───────────────────────────────────────────────────

    def _load_checkpoints(self):
        """Initialise shared encoder and task heads from saved checkpoints.

        The shared encoder receives the mean of all available encoder
        state dicts. Each task head is loaded from its own checkpoint.
        Checkpoints are expected to be present in the 'checkpoints/' directory.
        """
        device = next(self.parameters()).device
        encoder_state_dicts = []

        # ---- Classifier checkpoint ----
        if os.path.exists(_CKPT_CLASSIFIER):
            sd = _load_state_dict(_CKPT_CLASSIFIER, device)

            enc_sd = {k[len("encoder."):]: v
                      for k, v in sd.items() if k.startswith("encoder.")}
            encoder_state_dicts.append(enc_sd)

            cls_sd = {k[len("classifier."):]: v
                      for k, v in sd.items() if k.startswith("classifier.")}
            self.classifier.load_state_dict(cls_sd)
            print(f"[MultiTask] Loaded classifier head from {_CKPT_CLASSIFIER}")
        else:
            print(f"[MultiTask] WARNING: {_CKPT_CLASSIFIER} not found.")

        # ---- Localizer checkpoint ----
        if os.path.exists(_CKPT_LOCALIZER):
            sd = _load_state_dict(_CKPT_LOCALIZER, device)

            enc_sd = {k[len("encoder."):]: v
                      for k, v in sd.items() if k.startswith("encoder.")}
            encoder_state_dicts.append(enc_sd)

            loc_sd = {k[len("regressor."):]: v
                      for k, v in sd.items() if k.startswith("regressor.")}
            self.localizer.load_state_dict(loc_sd)
            print(f"[MultiTask] Loaded localizer head from {_CKPT_LOCALIZER}")
        else:
            print(f"[MultiTask] WARNING: {_CKPT_LOCALIZER} not found.")

        # ---- UNet checkpoint ----
        if os.path.exists(_CKPT_UNET):
            sd = _load_state_dict(_CKPT_UNET, device)

            enc_sd = {k[len("encoder."):]: v
                      for k, v in sd.items() if k.startswith("encoder.")}
            encoder_state_dicts.append(enc_sd)

            for prefix in ("dec4", "dec3", "dec2", "dec1", "dec0", "seg_head"):
                block_sd = {k[len(prefix) + 1:]: v
                            for k, v in sd.items() if k.startswith(prefix + ".")}
                getattr(self, prefix).load_state_dict(block_sd)
            print(f"[MultiTask] Loaded UNet decoder from {_CKPT_UNET}")
        else:
            print(f"[MultiTask] WARNING: {_CKPT_UNET} not found.")

        # ---- Average encoder weights across all available checkpoints ----
        if encoder_state_dicts:
            avg_enc_sd = {}
            for key in encoder_state_dicts[0]:
                stacked = torch.stack(
                    [sd[key].float() for sd in encoder_state_dicts], dim=0
                )
                avg_enc_sd[key] = stacked.mean(dim=0)
            self.encoder.load_state_dict(avg_enc_sd)
            print(f"[MultiTask] Shared encoder initialised from "
                  f"{len(encoder_state_dicts)} checkpoint(s).")

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """Single forward pass — encoder runs exactly once.

        Args:
            x: [B, 3, 224, 224]

        Returns:
            dict with keys:
                "classification" → [B, num_breeds]
                "localization"   → [B, 4]  pixel-space (cx, cy, w, h)
                "segmentation"   → [B, seg_classes, 224, 224]
        """
        bottleneck, features = self.encoder(x, return_features=True)

        f1 = features["block1"]
        f2 = features["block2"]
        f3 = features["block3"]
        f4 = features["block4"]
        f5 = features["block5"]

        flat = torch.flatten(bottleneck, 1)

        cls_out = self.classifier(flat)
        loc_out = self.localizer(flat)

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

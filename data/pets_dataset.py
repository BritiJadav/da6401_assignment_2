import os
from typing import Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T


# Input size fixed per VGG paper
_IMAGE_SIZE = 224


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Returns per sample:
        image : [3, 224, 224]  float32, ImageNet-normalised
        label : scalar         int64,   0-based breed index
        bbox  : [4]            float32, (cx, cy, w, h) in PIXEL SPACE
        mask  : [224, 224]     int64,   values in {0, 1, 2}
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = _IMAGE_SIZE,
        transform=None,
    ):
        """
        Args:
            root_dir:   Root directory of the Oxford-IIIT Pet dataset.
            split:      'train' or 'val'.
            image_size: Resize target (square). Default 224 per VGG paper.
            transform:  Optional torchvision transform applied to the PIL image.
                        If None, ToTensor + ImageNet normalisation is used.
        """
        self.root_dir   = root_dir
        self.split      = split
        self.image_size = image_size
        self.transform  = transform

        # ---- Paths ----
        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir  = os.path.join(root_dir, "annotations", "trimaps")

        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root_dir, "annotations", split_file)

        # ---- Read file list ----
        self.samples = []
        with open(split_path, "r") as f:
            for line in f:
                parts  = line.strip().split()
                img_id = parts[0]
                label  = int(parts[1]) - 1      # 0-based class index
                self.samples.append((img_id, label))

        # ---- Default transform: ToTensor + ImageNet normalisation ----
        self._default_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id, label = self.samples[idx]

        # ---- Load image ----
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        image    = Image.open(img_path).convert("RGB")

        # ---- Load segmentation mask ----
        mask_path = os.path.join(self.masks_dir, f"{img_id}.png")
        mask      = Image.open(mask_path)

        # ---- Resize ----
        image = T.Resize((self.image_size, self.image_size))(image)
        mask  = T.Resize(
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST,
        )(mask)

        # ---- Mask → tensor, shift {1,2,3} → {0,1,2} ----
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1

        # ---- Bounding box from mask — PIXEL SPACE (not normalised) ----
        ys, xs = torch.where(mask > 0)

        if len(xs) > 0 and len(ys) > 0:
            x_min = xs.min().float()
            x_max = xs.max().float()
            y_min = ys.min().float()
            y_max = ys.max().float()

            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width    = x_max - x_min
            height   = y_max - y_min
        else:
            # Fallback for empty masks
            x_center = torch.tensor(0.0)
            y_center = torch.tensor(0.0)
            width    = torch.tensor(0.0)
            height   = torch.tensor(0.0)

        # Pixel-space bbox — values in [0, image_size]
        bbox = torch.tensor(
            [x_center, y_center, width, height], dtype=torch.float32
        )

        # ---- Apply image transform ----
        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transform(image)

        return {
            "image": image,                          # [3, H, W]  float32
            "label": torch.tensor(label),            # scalar     int64
            "bbox":  bbox,                           # [4]        float32, pixel space
            "mask":  mask,                           # [H, W]     int64, {0,1,2}
        }

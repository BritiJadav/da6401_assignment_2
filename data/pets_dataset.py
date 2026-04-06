import os
from typing import Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 224,
        transform=None,
    ):
        """
        Args:
            root_dir:   root dataset directory
            split:      'train' or 'val'
            image_size: resize target (square)
            transform:  optional torchvision transforms applied to the PIL image
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = transform

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
                label  = int(parts[1]) - 1  # 0-based class index
                self.samples.append((img_id, label))

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
        resize_img  = T.Resize((self.image_size, self.image_size))
        resize_mask = T.Resize(
            (self.image_size, self.image_size),
            interpolation=Image.NEAREST   # nearest-neighbor keeps class indices intact
        )

        image = resize_img(image)
        mask  = resize_mask(mask)

        # ---- Mask → tensor, shift labels {1,2,3} → {0,1,2} ----
        mask = torch.from_numpy(np.array(mask)).long()
        mask = mask - 1

        # ---- Bounding box from mask (NORMALIZED to [0, 1]) ----
        ys, xs = torch.where(mask > 0)

        if len(xs) > 0 and len(ys) > 0:
            x_min = xs.min().float()
            x_max = xs.max().float()
            y_min = ys.min().float()
            y_max = ys.max().float()

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width    = x_max - x_min
            height   = y_max - y_min
        else:
            # Fallback for empty masks
            x_center = y_center = width = height = torch.tensor(0.0)

        # IMPORTANT: normalize to [0, 1] so IoU loss gets stable gradients
        bbox = torch.tensor(
            [x_center, y_center, width, height]
        ) / self.image_size

        # ---- Apply transforms ----
        if self.transform:
            image = self.transform(image)
        else:
            # Default: HWC numpy → CHW float tensor in [0, 1]
            image = (
                torch.from_numpy(np.array(image))
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

        return {
            "image": image,                      # [3, H, W]  float32
            "label": torch.tensor(label),        # scalar     int64
            "bbox":  bbox,                       # [4]        float32, normalized 0-1
            "mask":  mask,                       # [H, W]     int64,   values 0/1/2
        }

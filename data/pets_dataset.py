import os
import xml.etree.ElementTree as ET
from typing import Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T


_IMAGE_SIZE = 224


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Returns per sample:
        image : [3, 224, 224]  float32, ImageNet-normalised
        label : scalar         int64,   0-based breed index
        bbox  : [4]            float32, (cx, cy, w, h) in pixel space (scaled to 224x224)
        mask  : [224, 224]     int64,   values in {0, 1, 2}
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = _IMAGE_SIZE,
        transform=None,
    ):
        self.root_dir   = root_dir
        self.split      = split
        self.image_size = image_size
        self.transform  = transform

        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir  = os.path.join(root_dir, "annotations", "trimaps")
        self.xml_dir    = os.path.join(root_dir, "annotations", "xmls")

        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root_dir, "annotations", split_file)

        self.samples = []
        with open(split_path, "r") as f:
            for line in f:
                parts  = line.strip().split()
                img_id = parts[0]
                label  = int(parts[1]) - 1
                self.samples.append((img_id, label))

        self._default_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_bbox(self, img_id: str, orig_w: int, orig_h: int):
        """Read bbox from XML annotation and scale to image_size x image_size."""
        xml_path = os.path.join(self.xml_dir, f"{img_id}.xml")
        if not os.path.exists(xml_path):
            return None

        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj  = root.find("object")
        box  = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        # Convert to centre format
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w  = xmax - xmin
        h  = ymax - ymin

        # Scale to resized image space
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        cx = cx * scale_x
        cy = cy * scale_y
        w  = w  * scale_x
        h  = h  * scale_y

        return [cx, cy, w, h]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id, label = self.samples[idx]

        # ---- Load image ----
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        image    = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size  # PIL gives (width, height)

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

        # ---- Bounding box from XML annotation ----
        bbox_raw = self._load_bbox(img_id, orig_w, orig_h)
        if bbox_raw is not None:
            bbox = torch.tensor(bbox_raw, dtype=torch.float32)
        else:
            # Fallback: full image box
            bbox = torch.tensor([
                self.image_size / 2.0,
                self.image_size / 2.0,
                float(self.image_size),
                float(self.image_size),
            ], dtype=torch.float32)

        # ---- Apply image transform ----
        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transform(image)

        return {
            "image": image,
            "label": torch.tensor(label),
            "bbox":  bbox,
            "mask":  mask,
        }

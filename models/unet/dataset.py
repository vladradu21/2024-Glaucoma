import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class GlaucomaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)
        mask_name = image_name.replace(".jpg", ".bmp")
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
        mask[mask == 255.0] = 2  # white background
        mask[mask == 128] = 1  # gray optic disc
        mask[mask == 0] = 0  # black optic cup

        if self.transform is not None:
            augumentations = self.transform(image=image, mask=mask)
            image = augumentations["image"]
            mask = augumentations["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).long()

        return image, mask
    
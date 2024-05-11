import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from albumentations.pytorch import ToTensorV2

from models.unet.model import UNET
from models.unet.utils import load_checkpoint, apply_color_map

# Hyperparameters etc.
CURRENT_DIR = Path(__file__).parent
MODEL_PATH = CURRENT_DIR / 'checkpoint' / 'my_checkpoint.pth.tar'
INPUT_IMAGE_DIR = CURRENT_DIR.parent.parent / 'data' / 'predict'
OUTPUT_IMAGE_DIR = CURRENT_DIR.parent.parent / 'out' / 'predict'
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024


class Segmentation:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = UNET(in_channels=3, out_channels=3).to(self.device)

        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)

        load_checkpoint(torch.load(model_path, map_location=self.device), self.model)
        self.model.eval()

        self.transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255
            ),
            ToTensorV2()
        ])

    def segment(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        transformed = self.transform(image=image)
        image = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(image)
            preds = torch.argmax(preds, dim=1).unsqueeze(1)

        preds_color = apply_color_map(preds)
        return preds_color


def predict_mask(image_name):
    model_path = MODEL_PATH
    input_image_path = INPUT_IMAGE_DIR / image_name
    output_image_path = OUTPUT_IMAGE_DIR / image_name

    segmentation = Segmentation(model_path)
    prediction = segmentation.segment(str(input_image_path))
    torchvision.utils.save_image(prediction, str(output_image_path))


def main():
    if len(sys.argv) != 2:
        print('Usage: python segment.py <input_image_name>')
        return

    predict_mask(str(sys.argv[1]))


if __name__ == '__main__':
    main()

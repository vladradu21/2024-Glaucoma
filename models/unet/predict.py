import sys

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from albumentations.pytorch import ToTensorV2

from model import UNET
from utils import load_checkpoint, apply_color_map

# Hyperparameters etc.
MODEL_PATH = './checkpoint/my_checkpoint.pth.tar'
INPUT_IMAGE_DIR = '../../data/predict/'
OUTPUT_IMAGE_DIR = '../../out/predict/'
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024


class Predictor:
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

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        transformed = self.transform(image=image)
        image = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(image)
            preds = torch.argmax(preds, dim=1).unsqueeze(1)

        preds_color = apply_color_map(preds)
        return preds_color


def main():
    if len(sys.argv) != 2:
        print('Usage: python predict.py <input_image_name>')
        return

    model_path = MODEL_PATH
    input_image_path = INPUT_IMAGE_DIR + sys.argv[1]
    output_image_path = OUTPUT_IMAGE_DIR + sys.argv[1]

    predictor = Predictor(model_path)
    prediction = predictor.predict(input_image_path)
    torchvision.utils.save_image(prediction, output_image_path)


if __name__ == '__main__':
    main()

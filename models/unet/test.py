import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from model import UNET
from utils import (
    load_checkpoint,
    get_test_loader, check_accuracy, save_predictions_as_imgs
)

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
TEST_IMG_DIR = "../../datasets/refuge2/test/images/"
TEST_MASK_DIR = "../../datasets/refuge2/test/mask/"


def test_fn(loader, model):
    check_accuracy(loader, model, device=DEVICE)
    save_predictions_as_imgs(loader, model, folder="../../out/test/", mode="test", device=DEVICE)


def main():
    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    load_checkpoint(torch.load("checkpoint/my_checkpoint.pth.tar"), model)
    model.eval()

    test_loader = get_test_loader(TEST_IMG_DIR, TEST_MASK_DIR, BATCH_SIZE, test_transforms, NUM_WORKERS)
    test_fn(test_loader, model)


if __name__ == "__main__":
    main()

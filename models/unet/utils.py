import torch
import torchvision
from torch.utils.data import DataLoader

from models.unet.dataset import GlaucomaDataset


def save_checkpoint(state, filename='checkpoint/my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True
):
    train_ds = GlaucomaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = GlaucomaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_score / len(loader)}")


COLOR_MAP = {
    0: (0.0, 0.0, 0.0),  # negru
    1: (0.5, 0.5, 0.5),  # gri
    2: (1.0, 1.0, 1.0),  # alb
}


def apply_color_map(preds, num_classes=3):
    # Shape of predictions: [batch_size, 1, H, W] with values in [0, 1, 2]
    batch_size, _, H, W = preds.shape
    output = torch.zeros(batch_size, 3, H, W, device=preds.device)  # Ensure output is on the correct device
    for i in range(num_classes):
        mask = preds == i
        # Ensure color tensor is on the correct device
        color = torch.tensor(COLOR_MAP[i], device=preds.device).view(1, 3, 1, 1)
        output += mask.float() * color  # Apply color map to the mask
    return output


def save_predictions_as_imgs(loader, model, folder="../../out/valid/", mode="valid", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1).unsqueeze(1)

        preds_color = apply_color_map(preds)
        true_color = apply_color_map(y.unsqueeze(1))

        preds_color_with_padding = torch.nn.functional.pad(preds_color, pad=(1, 1, 1, 1), mode='constant', value=0)
        true_color_with_padding = torch.nn.functional.pad(true_color, pad=(1, 1, 1, 1), mode='constant', value=0)

        torchvision.utils.save_image(preds_color_with_padding, f"{folder}/{mode}_{idx}_pred.png")
        torchvision.utils.save_image(true_color_with_padding, f"{folder}/{mode}_{idx}_true.png")
    model.train()


def get_test_loader(
        test_dir,
        test_maskdir,
        batch_size,
        transform,
        num_workers,
        pin_memory=True
):
    test_ds = GlaucomaDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory
    )

    return test_loader

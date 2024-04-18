import os
import glob
from PIL import Image

# Hyperparameters etc.
SIZE = (512, 512)
BASE_PATH = os.path.abspath('../datasets/')
OUTPUT_PATH = os.path.abspath('../datasets/')

# Define the dataset structure
dataset_configurations = {
    "refuge2": {
        "test": {
            "images": "*.jpg",
            "mask": "*.bmp"
        },
        "train": {
            "images": "*.jpg",
            "mask": "*.bmp"
        },
        "valid": {
            "images": "*.jpg",
            "mask": "*.bmp"
        }
    }
}


def resize_images(source_path, target_path, pattern, size, is_mask=False):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    files = glob.glob(os.path.join(source_path, pattern))
    print(f"Resizing files from {source_path} save them to {target_path}")
    for file_path in files:
        with Image.open(file_path) as img:
            if is_mask:
                img = img.resize(size, Image.NEAREST)  # Use nearest neighbor interpolation for masks
            else:
                img = img.resize(size, Image.Resampling.LANCZOS)  # High-quality downsampling filter
            img.save(os.path.join(target_path, os.path.basename(file_path)))


def process_datasets(base_path, output_base):
    for dataset_name, configurations in dataset_configurations.items():
        for phase, phase_configs in configurations.items():
            for content_type, pattern in phase_configs.items():
                source_dir = os.path.join(base_path, dataset_name, phase, content_type)
                target_dir = os.path.join(output_base, f"{dataset_name}_resized", phase, content_type)
                is_mask = content_type == "mask"
                resize_images(source_dir, target_dir, pattern, SIZE, is_mask=is_mask)


def main():
    process_datasets(BASE_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()

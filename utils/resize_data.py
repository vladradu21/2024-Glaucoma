from PIL import Image
import os
import glob

SIZE = (512, 512)

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
            "mask": "*.png"
        }
    }
}

def resize_image(file_path, output_size):
    with Image.open(file_path) as img:
        img = img.resize(output_size, Image.Resampling.LANCZOS)
        if file_path.lower().endswith('.png'):
            # If the file is a .png, we want to change the path to .bmp
            file_path = file_path[:-4] + '.bmp'  # Change the extension
            mode = 'wb'  # We'll be writing a new file
        else:
            mode = 'r+b' if os.path.exists(file_path) else 'w+b'

        with open(file_path, mode) as f:
            if file_path.lower().endswith('.bmp'):
                img.save(f, format='BMP')
            else:
                img.save(f, format='JPEG')

        # After saving as BMP, if the original was PNG, delete the PNG file
        if mode == 'wb' and file_path.lower().endswith('.bmp'):
            os.remove(file_path[:-4] + '.png')

        print(f"Resized and saved {file_path}")

def process_dataset(dataset_root, config):
    for phase, types in config.items():
        for data_type, pattern in types.items():
            folder_path = os.path.join(dataset_root, phase, data_type)
            for file_path in glob.glob(os.path.join(folder_path, pattern)):
                resize_image(file_path, SIZE)

def main():
    dataset_root = "../datasets"
    for dataset_name, config in dataset_configurations.items():
        full_path = os.path.join(dataset_root, dataset_name)
        process_dataset(full_path, config)

if __name__ == "__main__":
    main()

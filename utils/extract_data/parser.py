import csv
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from metrics import (
    get_diagnosis,
    calculate_cup_to_disc_ratio,
    calculate_vcdr,
    calculate_hcdr,
    extract_roi,
    calculate_isnt_areas
)

# Hyperparameters etc.
TRAIN_MASK_DIR = '../../data/refuge2/train/mask'
CSV_TO_SAVE_DIR = '../../data/csv'


def parse_dir(directory, csv_save_path):
    image_files = [img for img in sorted(os.listdir(directory)) if img.lower().endswith('.bmp')]

    with open(csv_save_path, 'w', newline='') as csvfile:
        fieldnames = ['name', 'CDR', 'vCDR', 'hCDR', 'I', 'S', 'N', 'T', 'respectsISNT', 'NNR', 'diagnosis']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_name in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(directory, image_name)
            with Image.open(image_path) as image:
                image_array = np.array(image.convert('L'))
                roi = extract_roi(image_array)
                ratio = calculate_cup_to_disc_ratio(roi)
                vcdr = calculate_vcdr(roi)
                hcdr = calculate_hcdr(roi)
                i, s, n, t = calculate_isnt_areas(roi)
                isnt = True if i > s > n > t else False
                nnr = round((i + s) / (n + t), 3)
                diagnosis = get_diagnosis(image_name)

                writer.writerow({
                    'name': image_name,
                    'CDR': ratio,
                    'vCDR': vcdr,
                    'hCDR': hcdr,
                    'I': i,
                    'S': s,
                    'N': n,
                    'T': t,
                    'respectsISNT': isnt,
                    'NNR': nnr,
                    'diagnosis': diagnosis})

    print(f"Data has been written to {csv_save_path}")


def main():
    # Ensure the CSV directory exists
    os.makedirs(CSV_TO_SAVE_DIR, exist_ok=True)
    csv_file_path = os.path.join(CSV_TO_SAVE_DIR, 'metrics.csv')
    parse_dir(TRAIN_MASK_DIR, csv_file_path)


if __name__ == "__main__":
    main()

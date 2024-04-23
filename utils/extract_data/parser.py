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


def parse_dir(directory, csv_writer):
    image_files = [img for img in sorted(os.listdir(directory)) if img.lower().endswith('.bmp')]

    for image_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(directory, image_name)
        with Image.open(image_path) as image:
            image_array = np.array(image.convert('L'))
            roi = extract_roi(image_array)
            i, s, n, t = calculate_isnt_areas(roi)
            diagnosis = get_diagnosis(image_name)

            data = {
                'name': image_name,
                'CDR': calculate_cup_to_disc_ratio(roi),
                'vCDR': calculate_vcdr(roi),
                'hCDR': calculate_hcdr(roi),
                'I': i,
                'S': s,
                'N': n,
                'T': t,
                'respectsISNT': i > s > n > t,
                'NNR': round((i + s) / (n + t), 3),
                'diagnosis': diagnosis
            }
            csv_writer.writerow(data)


def main():
    # ensure the CSV directory exists
    os.makedirs(CSV_TO_SAVE_DIR, exist_ok=True)
    csv_file_path = os.path.join(CSV_TO_SAVE_DIR, 'metrics.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['name', 'CDR', 'vCDR', 'hCDR', 'I', 'S', 'N', 'T', 'respectsISNT', 'NNR', 'diagnosis']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        parse_dir(TRAIN_MASK_DIR, writer)


if __name__ == "__main__":
    main()

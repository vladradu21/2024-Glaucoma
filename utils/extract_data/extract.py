import csv
import os
import sys

import numpy as np
from PIL import Image

from draw import (
    h_cup_disc_indices,
    v_cup_disc_indices,
    draw_lines,
    draw_diagonals
)
from metrics import (
    extract_roi,
    calculate_cup_to_disc_ratio,
    calculate_vcdr,
    calculate_hcdr,
    calculate_isnt_areas
)

# Hyperparameters etc.
INPUT_IMAGE_DIR = '../../out/predict/'
OUTPUT_IMAGE_DIR = '../../out/roi/'
CSV_TO_SAVE_DIR = '../../out/csv'


def save_image_with_hcdr_vcdr_lines(roi, output_base_path):
    if roi is not None:
        # copy of the ROI
        roi_rgb_lines = Image.fromarray(roi).convert('RGB')

        # draw HCDR and VCDR lines
        h_indices = h_cup_disc_indices(roi)
        v_indices = v_cup_disc_indices(roi.T)
        roi_rgb_lines = draw_lines(roi_rgb_lines, h_indices, v_indices)

        # save the annotated image
        roi_rgb_lines.save(f"{output_base_path}_hvcdr.jpg")
        print(f"hCDR vCDR lines image saved to {output_base_path}_hvcdr.jpg")
    else:
        print("No ROI found.")


def save_image_with_diagonals(roi, output_base_path):
    if roi is not None:
        # copy of the ROI
        roi_rgb_diagonals = Image.fromarray(roi).convert('RGB')

        # draw diagonals
        roi_rgb_diagonals = draw_diagonals(roi_rgb_diagonals)

        # save the annotated image
        roi_rgb_diagonals.save(f"{output_base_path}_diags.jpg")
        print(f"Diagonal image saved to {output_base_path}_diags.jpg")
    else:
        print("No ROI found.")


def save_roi(roi, output_base_path):
    roi_rgb = Image.fromarray(roi).convert('RGB')
    roi_rgb.save(f"{output_base_path}_roi.jpg")
    print(f"ROI image saved to {output_base_path}")


def write_csv(data, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['name', 'CDR', 'vCDR', 'hCDR', 'I', 'S', 'N', 'T', 'respectsISNT', 'NNR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)
    print(f"Data has been written to {csv_path}")


def main():
    if len(sys.argv) != 2:
        print('Usage: python test.py <input_image_name>')
        return

    input_image_name = sys.argv[1]
    input_image_path = INPUT_IMAGE_DIR + input_image_name
    output_image_path = OUTPUT_IMAGE_DIR + input_image_name[:-4]

    image = Image.open(input_image_path).convert('L')
    image_array = np.array(image)
    roi = extract_roi(image_array)

    i, s, n, t = calculate_isnt_areas(roi)
    data = {
        'name': input_image_name,
        'CDR': calculate_cup_to_disc_ratio(roi),
        'vCDR': calculate_vcdr(roi),
        'hCDR': calculate_hcdr(roi),
        'I': i,
        'S': s,
        'N': n,
        'T': t,
        'respectsISNT': i > s > n > t,
        'NNR': round((i + s) / (n + t), 3)
    }
    csv_save_path = os.path.join(CSV_TO_SAVE_DIR, f"{input_image_name[:-4]}.csv")
    write_csv(data, csv_save_path)

    save_roi(roi, output_image_path)
    save_image_with_hcdr_vcdr_lines(roi, output_image_path)
    save_image_with_diagonals(roi, output_image_path)


if __name__ == '__main__':
    main()
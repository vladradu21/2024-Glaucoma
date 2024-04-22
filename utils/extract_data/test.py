import sys

import numpy as np
from PIL import Image

from metrics import extract_roi

INPUT_IMAGE_DIR = '../../out/predict/'
OUTPUT_IMAGE_DIR = '../../out/roi/'


def main():
    if len(sys.argv) != 2:
        print('Usage: python test.py <input_image_name>')
        return

    input_image_path = INPUT_IMAGE_DIR + sys.argv[1]
    output_image_path = OUTPUT_IMAGE_DIR + sys.argv[1]

    image = Image.open(input_image_path).convert('L')
    image_array = np.array(image)
    roi = extract_roi(image_array)

    if roi is not None:
        roi_image = Image.fromarray(roi)
        roi_image.save(output_image_path)
        print(f"ROI image saved to {output_image_path}")
    else:
        print("No ROI found")


if __name__ == '__main__':
    main()
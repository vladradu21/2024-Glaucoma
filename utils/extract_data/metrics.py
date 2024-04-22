import numpy as np

# Hyperparameters etc.
BLACK_THRESHOLD = 1
GRAY_THRESHOLD = 254


def get_diagnosis(image_name):
    return 'glaucoma' if image_name.lower().startswith('g') else 'healthy'


def calculate_cup_to_disc_ratio(image_array):
    cup_pixels = np.sum(image_array < BLACK_THRESHOLD)  # only black
    disc_pixels = np.sum(image_array < GRAY_THRESHOLD)  # black and gray pixels
    ratio = cup_pixels / disc_pixels if disc_pixels > 0 else 0
    return round(ratio, 3)


def calculate_max_seq_length(image_array, threshold):
    max_length = 0
    for line in image_array:
        changes = np.diff(np.concatenate(([0], line < threshold, [0])))

        # Start points / End points of sequences under the threshold
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]

        max_run_length = np.max(run_ends - run_starts) if run_ends.size > 0 else 0
        max_length = max(max_length, max_run_length)
    return max_length


def calculate_vcdr(image_array):
    # length of longest sequence of black pixels in a column
    v_cup_diameter = calculate_max_seq_length(image_array.T, BLACK_THRESHOLD)

    # length of longest sequence of black + gray in a column
    v_disc_diameter = calculate_max_seq_length(image_array.T, GRAY_THRESHOLD)

    ratio = v_cup_diameter / v_disc_diameter if v_disc_diameter > 0 else 0
    return round(ratio, 3)


def calculate_hcdr(image_array):
    # length of longest sequence of black pixels in a row
    h_cup_diameter = calculate_max_seq_length(image_array, BLACK_THRESHOLD)

    # length of longest sequence of black + gray pixels in a row
    h_disc_diameter = calculate_max_seq_length(image_array, GRAY_THRESHOLD)

    ratio = h_cup_diameter / h_disc_diameter if h_disc_diameter > 0 else 0
    return round(ratio, 3)


def extract_roi(image_array):
    # boundaries for optic disc
    indices = np.where(image_array <= GRAY_THRESHOLD)
    if not indices[0].size or not indices[1].size:
        return None

    # determine bounding box
    min_y, max_y = indices[0].min(), indices[0].max()
    min_x, max_x = indices[1].min(), indices[1].max()

    center_y, center_x = (max_y + min_y) // 2, (max_x + min_x) // 2
    size = max(max_y - min_y, max_x - min_x)
    size = size if size % 2 == 0 else size + 1

    # starting and ending coordinates of the ROI
    start_y = max(0, center_y - size // 2)
    start_x = max(0, center_x - size // 2)
    end_y = start_y + size
    end_x = start_x + size

    # necessary padding if the ROI extends beyond the original image
    pad_top = -start_y if start_y < 0 else 0
    pad_left = -start_x if start_x < 0 else 0
    pad_bottom = end_y - image_array.shape[0] if end_y > image_array.shape[0] else 0
    pad_right = end_x - image_array.shape[1] if end_x > image_array.shape[1] else 0

    # adjust the start and end points to remain within the image
    start_y = max(0, start_y)
    start_x = max(0, start_x)
    end_y = min(image_array.shape[0], end_y)
    end_x = min(image_array.shape[1], end_x)

    # extract the ROI and apply padding if necessary to maintain the square shape
    roi = image_array[start_y:end_y, start_x:end_x]
    roi = np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=255)

    return roi


def percent_of_gray_pixels(region, total_gray):
    sum_region = np.sum((region > BLACK_THRESHOLD) & (region < GRAY_THRESHOLD))
    return round((sum_region / total_gray), 3)


def calculate_isnt_areas(roi):
    if roi is None:
        return None

    size = roi.shape[0]
    x, y = np.ogrid[:size, :size]

    # diagonal rules
    above_main_diag = x < y
    below_main_diag = x > y
    above_secondary_diag = x + y < size - 1
    below_secondary_diag = x + y > size - 1

    # define the regions
    superior = roi[above_main_diag & above_secondary_diag]
    inferior = roi[below_main_diag & below_secondary_diag]
    nasal = roi[below_main_diag & above_secondary_diag]
    temporal = roi[above_main_diag & below_secondary_diag]

    # percent of gray pixels
    total_gray = np.sum((roi > BLACK_THRESHOLD) & (roi < GRAY_THRESHOLD))
    inferior_gray = percent_of_gray_pixels(inferior, total_gray)
    superior_gray = percent_of_gray_pixels(superior, total_gray)
    nasal_gray = percent_of_gray_pixels(nasal, total_gray)
    temporal_gray = percent_of_gray_pixels(temporal, total_gray)

    return inferior_gray, superior_gray, nasal_gray, temporal_gray

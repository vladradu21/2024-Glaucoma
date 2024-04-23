import numpy as np
from PIL import ImageDraw

# Hyperparameters etc.
BLACK_THRESHOLD = 1
GRAY_THRESHOLD = 254


def draw_diagonals(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # draw primary diagonal
    draw.line((0, 0, width, height), fill='brown', width=2)

    # draw secondary diagonal
    draw.line((width, 0, 0, height), fill='brown', width=2)

    return image


def draw_lines(image, h_indices, v_indices):
    draw = ImageDraw.Draw(image)

    # draw horizontal lines
    for ((x_start, y), (x_end, _)) in h_indices:
        draw.line((x_start, y, x_end, y), fill='purple', width=2)

    # draw vertical lines
    for ((y_start, x), (y_end, x)) in v_indices:
        draw.line((x, y_start, x, y_end), fill='green', width=2)
    return image


def max_seq_indices(image_array, threshold):
    max_length = 0
    start_index = 0  # starting index of the max sequence
    end_index = 0    # ending index of the max sequence
    max_line_index = 0  # the line index where max sequence occurs

    for line_index, line in enumerate(image_array):

        # detect changes from background to black/gray and vice versa
        changes = np.diff(np.concatenate(([0], line < threshold, [0])))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]

        # update the max sequence info if a longer sequence is found
        if run_ends.size > 0:
            max_run_length = np.max(run_ends - run_starts)
            if max_run_length > max_length:
                max_length = max_run_length
                max_run_index = np.argmax(run_ends - run_starts)
                start_index = run_starts[max_run_index]
                end_index = run_ends[max_run_index]
                max_line_index = line_index

    # tuple for start and end coordinates
    return (start_index, max_line_index), (end_index, max_line_index)  # (x_start, y, x_end, y) un-transposed array


def h_cup_disc_indices(image_array):
    h_cup_line = max_seq_indices(image_array, BLACK_THRESHOLD)
    h_disc_line = max_seq_indices(image_array, GRAY_THRESHOLD)
    return h_cup_line, h_disc_line   # (x_start, y, x_end, y), (x_start, y, x_end, y)


def v_cup_disc_indices(transposed_image_array):
    v_cup_line = max_seq_indices(transposed_image_array, BLACK_THRESHOLD)
    v_disc_line = max_seq_indices(transposed_image_array, GRAY_THRESHOLD)
    return v_cup_line, v_disc_line  # (y_start, x, y_end, x), (y_start, x, y_end, x)

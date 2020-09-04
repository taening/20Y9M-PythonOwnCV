import numpy as np
import cv2

THRESH_BINARY = 0
THRESH_BINARY_INV = 1
THRESH_TRUNC = 2
THRESH_TOZERO = 3
THRESH_TOZERO_INV = 4


def threshold(src, thresh, max_val, type):
    if not isinstance(src, np.ndarray):
        raise TypeError("Not ndarry Type")

    if isinstance(thresh, int):
        pass
    elif isinstance(thresh, float):
        thresh = round(thresh)
    else:
        raise TypeError("Wrong Type")

    if max_val > 255:
        max_val = 255
    if isinstance(max_val, int):
        pass
    elif isinstance(max_val, float):
        max_val = round(max_val)
    else:
        raise TypeError("Wrong Type")

    dst = np.zeros(src.shape, np.uint8)
    if len(src) == 3:
        row, col, channel = src.shape
    else:
        row, col = src.shape

    for r in range(row):
        for c in range(col):
            if src[r][c] > thresh:
                if type == THRESH_BINARY:
                    dst[r][c] = max_val
                elif type == THRESH_BINARY_INV:
                    dst[r][c] = 0
                elif type == THRESH_TRUNC:
                    dst[r][c] = thresh
                elif type == THRESH_TOZERO:
                    dst[r][c] = src[r][c]
                elif type == THRESH_TOZERO_INV:
                    dst[r][c] = 0
                else:
                    raise ValueError("Wrong Value")
            else:
                if type == THRESH_BINARY:
                    dst[r][c] = 0
                elif type == THRESH_BINARY_INV:
                    dst[r][c] = max_val
                elif type == THRESH_TRUNC:
                    dst[r][c] = src[r][c]
                elif type == THRESH_TOZERO:
                    dst[r][c] = 0
                elif type == THRESH_TOZERO_INV:
                    dst[r][c] = src[r][c]
                else:
                    raise ValueError("Wrong Value")
    return dst


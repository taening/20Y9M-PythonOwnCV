import numpy as np

MORPH_OPEN = 0
MORPH_CLOSE = 1
MORPH_GRADIENT = 2
MORPH_TOPHAT = 3
MORPH_BLACKHAT = 4


def erode(src, kernel):
    if not isinstance(src, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise TypeError("Can use ndarray type only")
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Set kernel odd number")
    if len(src.shape) != 2:
        raise TypeError("Can use binary image only")

    row, col = src.shape
    k_row, k_col = kernel.shape
    k_center = (k_row//2, k_col//2)
    k_num = np.count_nonzero(kernel)
    dst = np.zeros(src.shape, np.uint8)
    count = 0
    for r in range(row):
        for c in range(col):
            for k_r in range(-k_center[0], k_row-k_center[0]):
                for k_c in range(-k_center[1], k_col-k_center[1]):
                    if r + k_r < 0 or c + k_c < 0 or r + k_r > row - 1 or c + k_c > col - 1:
                        continue
                    if src[r + k_r][c + k_c] != 0 and kernel[k_r + k_center[0]][k_c + k_center[1]] == 1:
                        count += 1
            if count == k_num:
                dst[r][c] = src[r][c]
            else:
                dst[r][c] = 0
            count = 0
    return dst


def dilate(src, kernel):
    if not isinstance(src, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise TypeError("Can use ndarray type only")
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Set kernel odd number")
    if len(src.shape) != 2:
        raise TypeError("Can use binary image only")

    row, col = src.shape
    k_row, k_col = kernel.shape
    k_center = (k_row//2, k_col//2)
    dst = np.zeros(src.shape, np.uint8)
    for r in range(row):
        for c in range(col):
            for k_r in range(-k_center[0], k_row-k_center[0]):
                for k_c in range(-k_center[1], k_col-k_center[1]):
                    if r + k_r < 0 or c + k_c < 0 or r + k_r > row - 1 or c + k_c > col - 1:
                        continue
                    if src[r + k_r][c + k_c] != 0 and kernel[k_r + k_center[0]][k_c + k_center[1]] == 1:
                        dst[r][c] = src[r + k_r][c + k_c]
                        break
    return dst


def morphology(src, op, kernel):
    if op == MORPH_OPEN:
        dst = erode(src, kernel)
        dst = dilate(dst, kernel)
    elif op == MORPH_CLOSE:
        dst = dilate(src, kernel)
        dst = erode(dst, kernel)
    elif op == MORPH_GRADIENT:
        dst = dilate(src, kernel) - erode(src, kernel)
    elif op == MORPH_TOPHAT:
        dst = src - morphology(src, MORPH_OPEN, kernel)
    elif op == MORPH_BLACKHAT:
        dst = morphology(src, MORPH_CLOSE, kernel)
    else:
        raise TypeError("Flag Error")
    return dst


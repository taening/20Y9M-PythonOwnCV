import numpy as np


COLOR_BGR2GRAY = 0  # OK
COLOR_RGB2GRAY = 1  # OK
COLOR_BGR2HSV = 2  # OK
COLOR_RGB2HSV = 3  # OK
COLOR_HSV2BGR = 4  # Not Yet
COLOR_HSV2RGB = 5  # Not Yet
COLOR_GRAY2BGR = 6  # OK
COLOR_GRAY2RGB = 7  # OK


def cvt_color(src, code):
    if not isinstance(src, np.ndarray):
        raise TypeError

    dst = None
    if code == COLOR_BGR2GRAY or code == COLOR_RGB2GRAY:
        if code == COLOR_BGR2GRAY:
            b = src[:, :, 0]
            g = src[:, :, 1]
            r = src[:, :, 2]
        else:
            r = src[:, :, 0]
            g = src[:, :, 1]
            b = src[:, :, 2]
        dst = (0.299 * r) + (0.587 * g) + (0.114 * b)
        dst = dst.astype(np.uint8)

    elif code == COLOR_BGR2HSV or code == COLOR_RGB2HSV:
        if code == COLOR_BGR2HSV:
            b = src[:, :, 0]
            g = src[:, :, 1]
            r = src[:, :, 2]
        else:
            r = src[:, :, 0]
            g = src[:, :, 1]
            b = src[:, :, 2]
        c_max_map = np.argmax(src, axis=2)
        c_max = np.max(src, axis=2)
        c_min = np.min(src, axis=2)
        delta = c_max - c_min

        with np.errstate(all="ignore"):
            r_ = (((c_max - r / 255.0) / 6) + (delta / 2)) / delta
            g_ = (((c_max - g / 255.0) / 6) + (delta / 2)) / delta
            b_ = (((c_max - b / 255.0) / 6) + (delta / 2)) / delta

            conditions = [c_max_map == 0, c_max_map == 1, c_max_map == 2]
            choices = [(2 / 3) + g_ - r_, (1 / 3) + r_ - b_, b_ - g_]
            h = np.select(conditions, choices)

            conditions = [h < 0, (h >= 0) & (h <= 1), h > 1]
            choices = [h + 1, h, h - 1]

            h = np.select(conditions, choices) * 180.0
            s = delta / c_max * 255.0
            v = c_max
        dst = np.stack([h, s, v], axis=2).astype(np.uint8)

    elif code == COLOR_HSV2BGR or code == COLOR_HSV2RGB:
        if code == COLOR_HSV2BGR:
            pass
        elif code == COLOR_HSV2RGB:
            pass

    elif code == COLOR_GRAY2BGR or code == COLOR_GRAY2RGB:
        if len(src.shape) > 2:
            raise ValueError("Invalid number of channels in input image")
        dst = np.stack([src, src, src], axis=2).astype(np.uint8)

    else:
        raise ValueError
    return dst

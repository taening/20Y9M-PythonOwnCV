import numpy as np


COLOR_BGR2GRAY = 0  # OK
COLOR_RGB2GRAY = 1  # OK
COLOR_BGR2HSV = 2  # OK
COLOR_RGB2HSV = 3  # OK
COLOR_HSV2BGR = 4  # OK
COLOR_HSV2RGB = 5  # OK
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
        src_ = src / 255.0
        c_max_map = np.argmax(src_, axis=2)
        c_max = np.max(src_, axis=2)
        c_min = np.min(src_, axis=2)
        delta = c_max - c_min

        with np.errstate(all="ignore"):
            if code == COLOR_BGR2HSV:
                b_ = src_[:, :, 0]
                g_ = src_[:, :, 1]
                r_ = src_[:, :, 2]
                choices = [0, 60 * (((r_ - g_) / delta) + 4), 60 * (((b_ - r_) / delta) + 2),
                           60 * (((g_ - b_) / delta) % 6)]
            else:
                r_ = src_[:, :, 0]
                g_ = src_[:, :, 1]
                b_ = src_[:, :, 2]
                choices = [0, 60 * (((g_ - b_) / delta) % 6), 60 * (((b_ - r_) / delta) + 2),
                           60 * (((r_ - g_) / delta) + 4)]
            conditions = [delta == 0, c_max_map == 0, c_max_map == 1, c_max_map == 2]
            h = np.select(conditions, choices) / 2.0
            s = np.where(c_max == 0, 0, delta / c_max) * 255.0
            v = c_max * 255.0
        dst = np.stack([h, s, v], axis=2).astype(np.uint8)

    elif code == COLOR_HSV2BGR or code == COLOR_HSV2RGB:
        h = src[:, :, 0] * 2.0
        s = src[:, :, 1] / 255.0
        v = src[:, :, 2] / 255.0

        c = v * s
        x = c * (1.0 - np.abs((((h / 60.0) % 2.0) - 1.0)))
        m = v - c
        zero = np.zeros((src.shape[0], src.shape[1]), np.float64)

        conditions = [(h >= 0) & (h < 60), (h >= 60) & (h < 120), (h >= 120) & (h < 180),
                      (h >= 180) & (h < 240), (h >= 240) & (h < 300), (h >= 300) & (h < 360)]
        choices = [[c, x, zero], [x, c, zero], [zero, c, x], [zero, x, c], [x, zero, c], [c, zero, x]]
        rgb_ = np.select(conditions, choices)

        r = (rgb_[0] + m) * 255.0
        g = (rgb_[1] + m) * 255.0
        b = (rgb_[2] + m) * 255.0

        if code == COLOR_HSV2BGR:
            dst = np.stack([b, g, r], axis=2).astype(np.uint8)
        else:
            dst = np.stack([r, g, b], axis=2).astype(np.uint8)

    elif code == COLOR_GRAY2BGR or code == COLOR_GRAY2RGB:
        if len(src.shape) > 2:
            raise ValueError("Invalid number of channels in input image")
        dst = np.stack([src, src, src], axis=2).astype(np.uint8)

    else:
        raise ValueError
    return dst

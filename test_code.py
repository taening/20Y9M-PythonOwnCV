import numpy as np
import os
import cv2
import time as t

lst = []

# Path of test image
img_path = os.path.join(os.getcwd(), "data", "6.jpg")

# RGB image
ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
lst.append(ori_img)

# TODO:

# Show image
for i, l in enumerate(lst):
    cv2.imshow(str(i), l)
cv2.waitKey(0)

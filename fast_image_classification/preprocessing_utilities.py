import cv2
import numpy as np


def read_img_from_path(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def read_from_file(file_object):
    arr = np.fromstring(file_object.read(), np.uint8)
    img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return img_np


def resize_img(img, h=128, w=128):
    desired_size_h = h
    desired_size_w = w

    old_size = img.shape[:2]

    ratio = min(desired_size_w / old_size[1], desired_size_h / old_size[0])

    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size_w - new_size[1]
    delta_h = desired_size_h - new_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return new_im

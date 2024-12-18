import cv2
import numpy as np
import os
import sys


def bitwise_or(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:

    if image_1.dtype != np.uint8:

        image_1: np.ndarray = image_1.astype(np.uint8)

    if image_2.dtype != np.uint8:

        image_2: np.ndarray = image_2.astype(np.uint8)

    height_1, width_1 = image_1.shape[:2]
    height_2, width_2 = image_2.shape[:2]

    if height_1 != height_2 or width_1 != width_2:
    
        image_2: np.ndarray = cv2.resize(
            image_2,
            dsize=(width_1, height_1),
            interpolation=cv2.INTER_LANCZOS4
        )

    image_bitwise: np.ndarray = cv2.bitwise_or(
        image_1,
        image_2,
    )

    return image_bitwise

def dilate(image: np.ndarray, empty: bool=False, iterations: int=3,
           kernel_size: tuple=(3, 1), max_value: int=1, thresh: int=128,
           type: int=cv2.THRESH_BINARY_INV) -> tuple[list, bool]:

    dst: np.ndarray = threshold(
        image,
        max_value,
        thresh,
        type,
    )
    kernel: np.ndarray = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        kernel_size,
    )
    dilated: np.ndarray = cv2.dilate(
        dst,
        kernel,
        iterations=iterations,
    )
    contours, hierarchy = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    coordinates: list = list()

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        coordinates.append((x, y, x + w, y + h))

    if not coordinates and not empty:

        print(f"{sys._getframe(0).f_code.co_name} - Empty.")

        coordinates, empty = dilate(
            cv2.bitwise_not(image),
            True,
            iterations,
            kernel_size,
            1,
            thresh // 2,
            type,
        )

    return coordinates, empty

def threshold(image: np.ndarray, max_value: int=1, thresh: int=128,
              type: int=cv2.THRESH_BINARY_INV) -> np.ndarray:

    if len(image.shape) == 3 and image.shape[2] == 3:

        gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    else:

        gray: np.ndarray = image.copy()

    _, dst = cv2.threshold(
        gray,
        thresh,
        max_value,
        type,
    )

    return dst
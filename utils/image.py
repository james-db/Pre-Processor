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

def crop_image(coordinates: tuple, image: np.ndarray, border: int=0,
               color: tuple=(255, 255, 255),
               offset: int=0) -> tuple[np.ndarray, tuple]:

    if isinstance(coordinates, list):

        coordinates_new: list = []
        image_new: list = []

        while coordinates:

            coordinates_old: tuple = coordinates.pop(0)
            img_new, coor_new = crop_image(
                coordinates_old,
                image,
                border,
                color,
                offset,
            )
            coordinates_new.append(coor_new)
            image_new.append(img_new)

    elif isinstance(coordinates, tuple):

        x_1, y_1, x_2, y_2 = coordinates

        if offset:

            if len(image.shape) == 3:

                h, w, _ = image.shape

            elif len(image.shape) == 2:

                h, w, = image.shape

            x_1: int = max(0, x_1 - offset)
            x_2: int = min(w, x_2 + offset)
            y_1: int = max(0, y_1 - offset)
            y_2: int = min(h, y_2 + offset)

        coordinates_new: tuple = (x_1, y_1, x_2, y_2)
        image_new: np.ndarray = image[y_1:y_2, x_1:x_2]

        if border:

            image_new: np.ndarray = cv2.copyMakeBorder(
                image_new,
                border,
                border,
                border,
                border,
                cv2.BORDER_CONSTANT,
                value=color,
            )

    return image_new, coordinates_new

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

    coordinates: list = []

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

def imread(filename: str) -> np.ndarray:

    try:

        image: np.ndarray = cv2.imread(filename)
        
        if isinstance(image, type(None)):
            
            image: np.ndarray = cv2.imdecode(
                np.fromfile(filename, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

    except:

        print(f"{sys._getframe(0).f_code.co_name} - Can not read file. Filename : {filename}")

    return image

def imshow(image: np.ndarray, name: str):

    x: int = 0  # "linux".
    platform: str = sys.platform

    if platform == "win32":

        x: int = -15

    cv2.namedWindow(name)
    cv2.moveWindow(name, x, 0)
    cv2.imshow(name, image)

    while True:

        key = cv2.waitKey(0) & 0xFF

        if key == 27:

            cv2.destroyAllWindows()

            break

def imwrite(fname: str, src: np.array, params=None) -> bool:

    ext: str = os.path.splitext(fname)[1]
    retval, src = cv2.imencode(ext, src, params)

    if retval:

        with open(fname, "w+b") as f:

            src.tofile(f)

    else:

        print(f"{sys._getframe(0).f_code.co_name} - Can not wrtie file. Filename : {fname}")

    return retval

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
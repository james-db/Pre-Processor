import cv2
import matplotlib.colors as mcolors
import numpy as np
import random
import os
import sys


def get_random_color() -> tuple[int, int, int]:

    colors: dict = dict(
        mcolors.TABLEAU_COLORS
    )
    color_names: list = list(colors.keys())
    color_name: str = random.choice(color_names)
    color: tuple[float, float, float] = mcolors.to_rgb(colors[color_name])
    color: tuple[int, int, int] = tuple([int(255 * v) for v in color])
    
    return color

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
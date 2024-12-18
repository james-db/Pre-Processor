import copy
import gdown
import os
import sys


def download_from_googledrive(fname: str, id: str):

    if os.path.exists(fname):

        print(f"{sys._getframe(0).f_code.co_name} - File exist, pass.")

    else:

        print(f"{sys._getframe(0).f_code.co_name} - File download.")

        dir: str = os.path.dirname(fname)
        os.makedirs(dir, exist_ok=True)
        gdown.download(id=id, output=fname, verify=False)

def sort_coordinates(coordinates: list, width: int,
                     tolerance_factor: int=10) -> list:

    def _sort(coordinates, width: int, tolerance_factor: int=10) -> int:

        return ((coordinates[1] // tolerance_factor) * tolerance_factor) \
                * width + coordinates[0]

    new_coordinates: list = copy.deepcopy(coordinates)
    new_coordinates.sort(key=lambda x: _sort(x, width, tolerance_factor))

    return new_coordinates
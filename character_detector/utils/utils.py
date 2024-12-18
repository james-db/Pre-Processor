import copy
import math
import sys


def merge_coordinates(coordinates: list, ignore: int=5) -> list:

    temp: tuple = coordinates[0]

    if isinstance(temp, list):

        new_coordinates: list = list()

        while coordinates:

            old_coordinates: tuple = coordinates.pop(0)
            new_coor: tuple = merge_coordinates(
                old_coordinates,
                ignore,
            )
            new_coordinates.append(new_coor)

    elif isinstance(temp, tuple):

        new_coordinates: list = list()

        old_coordinates: tuple = coordinates.pop(0)
        new_coordinates.append(old_coordinates)

        while coordinates:

            old_coordinates: tuple = coordinates.pop(0)
            new_coor: tuple = new_coordinates.pop(-1)
            x_1_1, y_1_1, x_1_2, y_1_2 = new_coor
            x_2_1, y_2_1, x_2_2, y_2_2 = old_coordinates
            is_x_2_1_under_x_1_2: bool = x_2_1 <= x_1_2
            is_under_ignore: bool = x_2_2 - x_2_1 <= ignore

            if is_x_2_1_under_x_1_2 or is_under_ignore:

                new_coordinates.append(
                    (
                        x_1_1,
                        min(y_1_1, y_2_1),
                        max(x_1_2, x_2_2),
                        max(y_1_2, y_2_2),
                    ),
                )

            else:

                new_coordinates.append(new_coor)
                new_coordinates.append(old_coordinates)

    else:

        ValueError("Coordinates's type must be list.")

    return new_coordinates

def merge_text(text: list) -> str:

    # print(f"{sys._getframe(0).f_code.co_name} - In : {text}.")

    new_text: list = list()

    for text_row in text:

        new_text_row: list = list()

        for text_word in text_row:

            new_text_word: list = list()

            for txt in text_word:

                new_text_word.append(txt)

            new_text_row.append("".join(new_text_word))

        new_text.append(" ".join(new_text_row))

    new_text: str = "\n".join(new_text)

    # print(f"{sys._getframe(0).f_code.co_name} - Out : {new_text}.")

    return new_text

def scale_coordinates(coordinates: tuple, dpi: int=72,
                      json_dpi: int=72) -> tuple[int, int, int, int]:

    if isinstance(coordinates, list):

        new_coordinates: list = list()

        while coordinates:

            old_coordinates: tuple = coordinates.pop(0)
            new_coor: tuple = scale_coordinates(
                old_coordinates,
                dpi,
                json_dpi,
            )
            new_coordinates.append(new_coor)

    if isinstance(coordinates, tuple):

        x_1, y_1, x_2, y_2 = coordinates
        x_1: int = int(math.floor(x_1 / json_dpi * dpi))
        x_2: int = int(math.ceil(x_2 / json_dpi * dpi))
        y_1: int = int(math.floor(y_1 / json_dpi * dpi))
        y_2: int = int(math.ceil(y_2 / json_dpi * dpi))
        new_coordinates: tuple = (x_1, y_1, x_2, y_2)

    else:
        
        ValueError("Coordinates's type must be list or tuple.")

    return new_coordinates

def shift_coordinates(coordinates: tuple,
                      origin: tuple=(0, 0)) -> tuple[int, int, int, int]:

    if isinstance(coordinates, list):

        new_coordinates: list = list()

        while coordinates:

            old_coordinates: tuple = coordinates.pop(0)
            new_coor: tuple = shift_coordinates(
                old_coordinates,
                origin,
            )
            new_coordinates.append(new_coor)

    elif isinstance(coordinates, tuple):

        x_0, y_0 = origin
        x_1, y_1, x_2, y_2 = coordinates
        new_coordinates: tuple = (
            x_0 + x_1,
            y_0 + y_1,
            x_0 + x_2,
            y_0 + y_2,
        )

    else:
        
        ValueError("Coordinates's type must be list or tuple.")

    return new_coordinates

def sort_coordinates(coordinates: list, width: int,
                     tolerance_factor: int=10) -> list:

    def _sort(coordinates, width: int, tolerance_factor: int=10) -> int:

        return ((coordinates[1] // tolerance_factor) * tolerance_factor) \
                * width + coordinates[0]

    new_coordinates: list = copy.deepcopy(coordinates)
    new_coordinates.sort(key=lambda x: _sort(x, width, tolerance_factor))

    return new_coordinates

def split_row(coordinates: list) -> list:

    new_coordinates: list = list()
    coordinates_row: list = list()
    x_pre: int = -1

    while coordinates:

        coor = coordinates.pop(0)
        x_cur: int = coor[0]

        if x_pre < x_cur:

            coordinates_row.append(coor)
            x_pre: int = x_cur

        else:

            coordinates.insert(0, coor)
            new_coordinates.append(copy.deepcopy(coordinates_row))
            coordinates_row.clear()
            x_pre: int = -1

    if coordinates_row:

        new_coordinates.append(coordinates_row)

    return new_coordinates

def split_space(coordinates: list, space: int=5) -> list:

    temp: tuple = coordinates[0]

    if isinstance(temp, list):

        new_coordinates: list = list()

        while coordinates:

            old_coordinates: tuple = coordinates.pop(0)
            new_coor: tuple = split_space(
                old_coordinates,
                space,
            )
            new_coordinates.append(new_coor)

    elif isinstance(temp, tuple):

        new_coordinates: list = list()
        coordinates_word: list = list()
        old_coordinates: tuple = coordinates.pop(0)
        coordinates_word.append(old_coordinates)

        while coordinates:

            old_coordinates: tuple = coordinates.pop(0)
            new_coor: tuple = coordinates_word.pop(-1)
            x_1_1, y_1_1, x_1_2, y_1_2 = new_coor
            x_2_1, y_2_1, x_2_2, y_2_2 = old_coordinates
            sub: int = x_2_1 - x_1_2

            if space <= sub:

                coordinates_word.append(new_coor)
                new_coordinates.append(copy.deepcopy(coordinates_word))
                coordinates_word.clear()
                coordinates_word.append(old_coordinates)

            else:

                coordinates_word.append(new_coor)
                coordinates_word.append(old_coordinates)

        if coordinates_word:

            new_coordinates.append(coordinates_word)

    return new_coordinates
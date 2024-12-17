import copy
import math
import sys


def merge_coordinates(coordinates: list, ignore: int=5) -> list:

    temp: tuple = coordinates[0]

    if isinstance(temp, list):

        coordinates_new: list = list()

        while coordinates:

            coordinates_old: tuple = coordinates.pop(0)
            coor_new: tuple = merge_coordinates(
                coordinates_old,
                ignore,
            )
            coordinates_new.append(coor_new)

    elif isinstance(temp, tuple):

        coordinates_new: list = list()

        coordinates_old: tuple = coordinates.pop(0)
        coordinates_new.append(coordinates_old)

        while coordinates:

            coordinates_old: tuple = coordinates.pop(0)
            coor_new: tuple = coordinates_new.pop(-1)
            x_1_1, y_1_1, x_1_2, y_1_2 = coor_new
            x_2_1, y_2_1, x_2_2, y_2_2 = coordinates_old
            is_x_2_1_under_x_1_2: bool = x_2_1 <= x_1_2
            is_under_ignore: bool = x_2_2 - x_2_1 <= ignore

            if is_x_2_1_under_x_1_2 or is_under_ignore:

                coordinates_new.append(
                    (
                        x_1_1,
                        min(y_1_1, y_2_1),
                        max(x_1_2, x_2_2),
                        max(y_1_2, y_2_2),
                    ),
                )

            else:

                coordinates_new.append(coor_new)
                coordinates_new.append(coordinates_old)

    else:

        ValueError("Coordinates's type must be list.")

    return coordinates_new

def merge_text(text: list) -> str:

    print(f"{sys._getframe(0).f_code.co_name} - In : {text}.")

    text_new: list = list()

    for text_row in text:

        text_row_new: list = list()

        for text_word in text_row:

            text_word_new: list = list()

            for txt in text_word:

                text_word_new.append(txt)

            text_row_new.append("".join(text_word_new))

        text_new.append(" ".join(text_row_new))

    text_new: str = "\n".join(text_new)

    print(f"{sys._getframe(0).f_code.co_name} - Out : {text_new}.")

    return text_new

def scale_coordinates(coordinates: tuple, dpi: int=72,
                      json_dpi: int=72) -> tuple[int, int, int, int]:

    if isinstance(coordinates, list):

        coordinates_new: list = list()

        while coordinates:

            coordinates_old: tuple = coordinates.pop(0)
            coor_new: tuple = scale_coordinates(
                coordinates_old,
                dpi,
                json_dpi,
            )
            coordinates_new.append(coor_new)

    if isinstance(coordinates, tuple):

        x_1, y_1, x_2, y_2 = coordinates
        x_1: int = int(math.floor(x_1 / json_dpi * dpi))
        x_2: int = int(math.ceil(x_2 / json_dpi * dpi))
        y_1: int = int(math.floor(y_1 / json_dpi * dpi))
        y_2: int = int(math.ceil(y_2 / json_dpi * dpi))
        coordinates_new: tuple = (x_1, y_1, x_2, y_2)

    else:
        
        ValueError("Coordinates's type must be list or tuple.")

    return coordinates_new

def shift_coordinates(coordinates: tuple,
                      origin: tuple=(0, 0)) -> tuple[int, int, int, int]:

    if isinstance(coordinates, list):

        coordinates_new: list = list()

        while coordinates:

            coordinates_old: tuple = coordinates.pop(0)
            coor_new: tuple = shift_coordinates(
                coordinates_old,
                origin,
            )
            coordinates_new.append(coor_new)

    elif isinstance(coordinates, tuple):

        x_0, y_0 = origin
        x_1, y_1, x_2, y_2 = coordinates
        coordinates_new: tuple = (
            x_0 + x_1,
            y_0 + y_1,
            x_0 + x_2,
            y_0 + y_2,
        )

    else:
        
        ValueError("Coordinates's type must be list or tuple.")

    return coordinates_new

def sort_coordinates(coordinates: list, width: int,
                     tolerance_factor: int=10) -> list:

    def _sort(coordinates, width: int, tolerance_factor: int=10) -> int:

        return ((coordinates[1] // tolerance_factor) * tolerance_factor) \
                * width + coordinates[0]

    coordinates.sort(key=lambda x: _sort(x, width, tolerance_factor))

    return coordinates

def split_row(coordinates: list) -> list:

    coordinates_new: list = list()
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
            coordinates_new.append(copy.deepcopy(coordinates_row))
            coordinates_row.clear()
            x_pre: int = -1

    if coordinates_row:

        coordinates_new.append(coordinates_row)

    return coordinates_new

def split_space(coordinates: list, space: int=5) -> list:

    temp: tuple = coordinates[0]

    if isinstance(temp, list):

        coordinates_new: list = list()

        while coordinates:

            coordinates_old: tuple = coordinates.pop(0)
            coor_new: tuple = split_space(
                coordinates_old,
                space,
            )
            coordinates_new.append(coor_new)

    elif isinstance(temp, tuple):

        coordinates_new: list = list()
        coordinates_word: list = list()
        coordinates_old: tuple = coordinates.pop(0)
        coordinates_word.append(coordinates_old)

        while coordinates:

            coordinates_old: tuple = coordinates.pop(0)
            coor_new: tuple = coordinates_word.pop(-1)
            x_1_1, y_1_1, x_1_2, y_1_2 = coor_new
            x_2_1, y_2_1, x_2_2, y_2_2 = coordinates_old
            sub: int = x_2_1 - x_1_2

            if space <= sub:

                coordinates_word.append(coor_new)
                coordinates_new.append(copy.deepcopy(coordinates_word))
                coordinates_word.clear()
                coordinates_word.append(coordinates_old)

            else:

                coordinates_word.append(coor_new)
                coordinates_word.append(coordinates_old)

        if coordinates_word:

            coordinates_new.append(coordinates_word)

    return coordinates_new
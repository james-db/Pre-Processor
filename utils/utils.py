import matplotlib.colors as mcolors
import random


def get_random_color() -> tuple[int, int, int]:

    colors: dict = dict(
        mcolors.TABLEAU_COLORS
    )
    color_names: list = list(colors.keys())
    color_name: str = random.choice(color_names)
    color: tuple[float, float, float] = mcolors.to_rgb(colors[color_name])
    color: tuple[int, int, int] = tuple([int(255 * v) for v in color])
    
    return color
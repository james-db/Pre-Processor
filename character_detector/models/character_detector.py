import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from collections import OrderedDict
import cv2
import gdown
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable


from .CRAFT_pytorch import imgproc
from .CRAFT_pytorch.craft import CRAFT
from character_detector.utils.image import (
    bitwise_or,
    dilate,
    threshold,
)
from character_detector.utils.utils import (
    merge_coordinates,
    sort_coordinates,
    split_row,
    split_space,
)
from utils.gpu import available_gpu
from utils.utils import download_from_googledrive


class Craft():

    def __init__(self, f: str=""):

        self.need_mem: int = 419430400

        if os.path.isfile(f):

            self.f: str = f

        else:

            print(f"{sys._getframe(0).f_code.co_name} - Model file not exist. Use default model.")

            cur_dir: str = os.path.abspath(os.path.dirname(__file__))
            self.f: str = os.path.join(
                cur_dir,
                "CRAFT_pytorch/model/craft_mlt_25k.pth",
            )
            download_from_googledrive(self.f, "1lx1A-pl0_bhvWPXiYm_qDXKl6gYPVkNG")

    def __call__(self, image: np.ndarray, canvas_size: int=1280,
                 mag_ratio: float=1.5) -> np.ndarray:

        x, ratio = self.__pre_process(image, canvas_size, mag_ratio)
        y: torch.Tensor = self.predict(x)
        heatmap: np.ndarray = y[0,:,:,0].cpu().data.numpy()
        heatmap: np.ndarray = self.__post_process(
            heatmap,
            image.shape[:2],
            ratio,
        )

        return heatmap

    def __pre_process(self, image: np.ndarray, canvas_size: int=1280,
                    mag_ratio: float=1.5) -> tuple[torch.Tensor, float]:

        image_resized, ratio, _ = imgproc.resize_aspect_ratio(
            image,
            canvas_size,
            interpolation=cv2.INTER_LINEAR,
            mag_ratio=mag_ratio,
        )
        x: np.ndarray = imgproc.normalizeMeanVariance(image_resized)
        x: torch.Tensor = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w].
        x: torch.Tensor = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w].

        return x, ratio

    def __post_process(self, heatmap: np.ndarray, shape: tuple[int, int],
                     ratio: float) -> np.ndarray:

        height, width = shape
        heatmap_rezided: np.ndarray = cv2.resize(
            heatmap,
            dsize=None,
            fx=(1 / ratio * 2),
            fy=(1 / ratio * 2),
            interpolation=cv2.INTER_LANCZOS4
        )
        heatmap: np.ndarray = heatmap_rezided[:height, :width]

        return heatmap

    def build(self) -> None:

        is_available_gpu, _ = available_gpu(self.need_mem)
        device: str = "cuda" if is_available_gpu else "cpu"

        print(f"{sys._getframe(0).f_code.co_name} - Character Detector use device : {device}.")
        print(f"{sys._getframe(0).f_code.co_name} - Character Detector use weights : {self.f}.")

        self.model: nn.Module = CRAFT()
        self.model.load_state_dict(
            copy_state_dict(
                torch.load(
                    self.f,
                    map_location=device
                ),
            ),
        )
        self.model.eval()

    def predict(self, x):

        with torch.no_grad():

            y, _ = self.model(x)

        return y

    def split_character(self, image: np.ndarray, canvas_size: int = 1280,
                        ignore: int=5, iterations: int=3,
                        kernel_size: tuple=(4, 1), mag_ratio: float=1.5,
                        offset: int=1, space: int=5,
                        text_threshold: float=0.4,
                        tolerance_factor: float=10) -> tuple[list, bool]:  # Test.

        heatmap: np.ndarray = self.__call__(
            image,
            canvas_size,
            mag_ratio,
        )
        heatmap_thresh: np.ndarray = threshold(
            heatmap,
            1,
            text_threshold,
            cv2.THRESH_BINARY,
        )
        _, empty = dilate(  # Check if text color is black.
            image,
            iterations=iterations,
            kernel_size=kernel_size,
        )
        image_thresh: np.ndarray = threshold(
            image if not empty else cv2.bitwise_not(image),
        )
        concat: np.ndarray = bitwise_or(
            image_thresh,
            heatmap_thresh,
        )
        contours, _ = cv2.findContours(
            concat,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        coordinates: list = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)
            coordinates.append((x, y, x + w, y + h))

        coordinates: list = sort_coordinates(
            coordinates,
            image.shape[1],
            tolerance_factor,
        )
        coordinates: list = split_row(coordinates)
        coordinates: list = merge_coordinates(coordinates, ignore)
        coordinates: list = split_space(coordinates, space)

        return coordinates, empty

@staticmethod
def copy_state_dict(state_dict: dict) -> dict:

    if list(state_dict.keys())[0].startswith("module"):

        start_idx: int = 1

    else:

        start_idx: int = 0

    new_state_dict: OrderedDict = OrderedDict()

    for k, v in state_dict.items():

        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v

    return new_state_dict
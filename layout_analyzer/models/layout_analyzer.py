import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import gdown
import math
import natsort
import numpy as np
import os
import sys
import tqdm
import yaml


from layout_analyzer.models.VGT.ditod.config import add_vit_config
from layout_analyzer.models.VGT.ditod.VGTTrainer import DefaultPredictor
from layout_analyzer.utils.utils import (
    get_sorted_indices,
    pdf2image,
    pdf2pickle,
)
from utils.gpu import available_gpu
from utils.image import imread


class VGT(DefaultPredictor):

    def __init__(self, config_path: str="", dataset: str="Doclaynet",
                 dpi: int = 72, model_path: str="",
                 wordgrid_model_path: str="",
                 tokenizer: str="google-bert/bert-base-uncased"):

        need_mem: int = 4613734400
        is_available_gpu, _ = available_gpu(need_mem)
        device: str = "cuda" if is_available_gpu else "cpu"  # Set device.

        is_exist_config: bool = os.path.isfile(config_path)
        is_exist_model: bool = os.path.isfile(model_path)
        is_exist_wordgrid_model: bool = os.path.isfile(wordgrid_model_path)

        self.cur_dir: str = os.path.abspath(os.path.dirname(__file__))
        config: dict = self.__get_setting(dataset)
        self.classes: list = config.get("classes")
        self.dpi: int = dpi
        self.tokenizer = tokenizer
        self.tolerance_factor: int = max(
            10,
            int(math.floor(self.dpi / 72 * 10)),
        )

        if not is_exist_config:

            print(f"{sys._getframe(0).f_code.co_name} - Config file not exist. Use default config.")

            config_fname: str = config.get("config")
            config_path: str = os.path.join(
                self.cur_dir,
                "VGT/configs/cascade",
                config_fname,
            )

        if not is_exist_model:

            print(f"{sys._getframe(0).f_code.co_name} - Model file not exist. Use default model.")

            model_fname: str = config.get("model")
            model_id: str = config.get("id")
            model_path: str = os.path.join(
                self.cur_dir,
                "VGT/model",
                model_fname,
            )
            self.__download(model_path, model_id)

        if not is_exist_wordgrid_model:

            print(f"{sys._getframe(0).f_code.co_name} - Word grid model file not exist. Use default model.")

            wordgrid: dict = config.get("wordgrid")
            wordgrid_model_fname: str = wordgrid.get("model")
            wordgrid_model_id: str = wordgrid.get("id")
            wordgrid_model_path = os.path.join(
                self.cur_dir,
                "VGT/model",
                wordgrid_model_fname,
            )
            self.__download(
                wordgrid_model_path,
                wordgrid_model_id,
            )

        cfg = get_cfg()  # Instantiate config.
        add_vit_config(cfg)
        cfg.merge_from_file(config_path)
        opts: list = [
            "MODEL.WEIGHTS",
            model_path,
            "MODEL.WORDGRID.MODEL_PATH",
            wordgrid_model_path,
        ]
        cfg.merge_from_list(opts)  # Add model weights URL to config.
        cfg.MODEL.DEVICE = device

        super().__init__(cfg)

    def __download(self, fname: str, id: str):

        if os.path.exists(fname):

            print(f"{sys._getframe(0).f_code.co_name} - File exist, pass. Filename : {fname}")

        else:

            print(f"{sys._getframe(0).f_code.co_name} - Download files.")

            dir: str = os.path.dirname(fname)
            os.makedirs(dir, exist_ok=True)
            gdown.download(id=id, output=fname, verify=False)

    def __get_setting(self, dataset: str) -> dict:

        setting_path: str = os.path.join(
                self.cur_dir,
                "VGT/configs/setting.yaml",
            )

        with open(setting_path, "r") as f:

            try:

                settings = yaml.safe_load(f)

            except yaml.constructor.ConstructorError:

                print(f"{sys._getframe(0).f_code.co_name} - Loading config {setting_path} with yaml.unsafe_load. Your machine may be at risk if the file contains malicious content.")

                f.close()

                with open(setting_path, "r") as f:

                    settings = yaml.unsafe_load(f)

        setting: dict = settings.get(dataset)

        return setting

    def analyze(self, fname: str) -> list:

        image_dir, pickle_dir = self.pre_process(fname)
        image_fnames: list = natsort.natsorted(os.listdir(image_dir))

        annotations: list = list()
        id: int = 1

        for i, image_fname in enumerate(tqdm.tqdm(image_fnames), 1):

            image: np.ndarray = imread(os.path.join(image_dir, image_fname))
            pickle_fname: str = os.path.join(
                pickle_dir,
                f"{os.path.splitext(image_fname)[0]}.pkl",
            )
            result: dict = self.__call__(image, pickle_fname)
            width: int = image.shape[1]
            annots, id = self.post_process(id, i, result, width)
            annotations.extend(annots)

            print(f"{sys._getframe(0).f_code.co_name} - annotations : {annotations}.")
            
            break

        return annotations

    def post_process(self, id: int, page: str, result: dict,
                     width: int) -> tuple[list, int]:

        instances = result.get("instances")
        boxes: list = instances.pred_boxes.tensor.cpu().numpy().tolist() \
            if instances.has("pred_boxes") else None
        categories: list = [
            self.classes[i] for i in instances.pred_classes.tolist()
        ] if instances.has("pred_classes") else None
        scores: list = instances.scores.tolist() \
            if instances.has("scores") else None
        indices: list = get_sorted_indices(
            boxes,
            width,
            self.tolerance_factor,
        )

        annotations: list = list()

        for i in indices:

            box: tuple = tuple([c / self.dpi for c in boxes[i]])
            annotation: dict = {
                "id": id,
                "box": box,
                "category": categories[i],
                "page": page,
                "score": scores[i],
            }
            annotations.append(annotation)
            id += 1

        return annotations, id

    def pre_process(self, fname: str) -> tuple[str, str]:

        print(f"{sys._getframe(0).f_code.co_name} - Pre-prcoessing. Filename : {fname}.")

        image_dir: str = pdf2image(fname, self.dpi)
        pickle_dir: str = pdf2pickle(fname, self.tokenizer)

        return image_dir, pickle_dir
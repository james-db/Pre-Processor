import easyocr
import numpy as np
import os
from paddleocr import PaddleOCR
import pytesseract
from surya.ocr import run_ocr
from surya.model.detection.model import (
    load_model as load_det_model,
    load_processor as load_det_processor,
)
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import sys


from utils.gpu import available_gpu


class EasyOCR():

    def __init__(self, language: list=["ko"]):

        need_mem: int = 524288000
        is_available_gpu, _ = available_gpu(need_mem)
        self.reader = easyocr.Reader(
            ["ko","en"],
            gpu=is_available_gpu,
        )
        self.language: list = language

        pass

    def __call__(self, image: np.ndarray) -> str:

        result: list = self.reader.recognize(
            image,
        )
        text: str = "\n".join([res[1] for res in result])

        # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")

        return text

    def recognize_character(self, image: list) -> str:

        temp: np.ndarray = image[0]

        if isinstance(temp, list):

            text: list = []

            while image:

                image_i: np.ndarray = image.pop(0)
                txt: str = self.recognize_character(
                    image_i,
                )

                text.append(txt)

        elif isinstance(temp, np.ndarray):

            text: list = []

            while image:

                image_i: np.ndarray = image.pop(0)
                txt: str = self.__call__(
                    image_i,
                )

                text.append(txt)

        else:

            ValueError("Image's type must be np.ndarray.")

        return text

class Paddle(PaddleOCR):

    def __init__(self, **kwargs):

        print(f"{sys._getframe(0).f_code.co_name} - kwargs : {kwargs}.")  # Test.

        super().__init__(**kwargs)

    def recognize_character(self, image: list, classify: bool=False,
                            detect: bool=False, recognize: bool=True) -> str:

        temp: np.ndarray = image[0]

        if isinstance(temp, list):

            text: list = []

            while image:

                image_i: np.ndarray = image.pop(0)
                txt: str = self.recognize_character(
                    image_i,
                    classify,
                    detect,
                    recognize,
                )

                text.append(txt)

        elif isinstance(temp, np.ndarray):

            text: list = []

            while image:

                image_i: np.ndarray = image.pop(0)
                result: str = self.ocr(
                    image_i,
                    detect,
                    recognize,
                    classify,
                )
                txt: str = "". join(
                    [r[0] for res in result for r in res],
                )

                text.append(txt)

        else:

            ValueError("Image's type must be np.ndarray.")

        return text

class Surya():

    def __init__(self, language: list=["en", "ko"]):

        need_mem: int = 1363148800
        is_available_gpu, _ = available_gpu(need_mem)
        device: str = "cuda" if is_available_gpu else "cpu"
        self.language = language
        self.detection_model = load_det_model(device=device)
        self.detection_processor = load_det_processor()
        self.recognition_model = load_rec_model(device=device)
        self.recognition_processor = load_rec_processor()

    def __call__(self, image: np.ndarray, detect: bool=True,
                 recognize: bool=True) -> list:

        detection_model = self.detection_model if detect else None
        detection_processor = self.detection_processor if detect else None
        recognition_model = self.recognition_model if recognize else None
        recognition_processor = self.recognition_processor if recognize else None

        result: list = run_ocr(
            [image],
            [self.language],
            detection_model,
            detection_processor,
            recognition_model,
            recognition_processor,
        )

        return result

    def recognize(self, image: np.ndarray, detect: bool=True,
                  recognize: bool=True) -> str:

        result: list = self.__call__(
            image,
            detect,
            recognize,
        )

        text: list = []

        for res in result:

            text.append(
                "".join([text_lines.text for text_lines in res.text_lines])
            )

        text: str = "\n".join(text)

        # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")

        return text

class Tesseract():

    def __init__(self, tesseract_cmd: str):

        self.__valid_tesseract_cmd(tesseract_cmd)
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.dict: dict = {}

    def __call__(self, image: np.ndarray, language: str,
                 config: str="") -> dict:

        data: dict = pytesseract.image_to_data(
            image,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        return data

    def __valid_text(self, language: str, text: str) -> bool:

        flag: bool = False
        lang_dict: dict = self.dict.get(language, [])

        if not lang_dict:

            # print(f"{sys._getframe(0).f_code.co_name} - Can not valid text. Becasue {language}'s dictionary is not exist.")

            flag: bool = True

        else:

            for character in text:

                if character in lang_dict:

                    flag: bool = True

                    break

        return flag

    def __valid_tesseract_cmd(self, tesseract_cmd) -> None:

        # tesseract_dir: str = os.path.dirname(tesseract_cmd)
        # is_exist_tesseract_dir: bool = os.path.exists(tesseract_dir)
        # is_dir_tesseract_dir: bool = os.path.isdir(tesseract_dir)

        platform: str = sys.platform

        # print(f"{sys._getframe(0).f_code.co_name} - Platform : {platform}.")

        if platform == "win32":

            tesseract_cmd: str = f"{tesseract_cmd}.exe"

        # is_exist_tesseract_cmd: bool = os.path.exists(tesseract_cmd)
        is_file_tesseract_cmd: bool = os.path.isfile(tesseract_cmd)

        if not is_file_tesseract_cmd:

            raise ValueError("Please use corrent Tesseract path.")

    def add_dictionary(self, language, dict_dir: str = "") -> None:

        is_dir_dict_dir: bool = os.path.isdir(dict_dir)

        if not is_dir_dict_dir:

            cur_dir: str = os.path.abspath(os.path.dirname(__file__))
            dict_dir: str = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "Tesseract/dict",
            )

        langs: list = language.split("+")

        # print(f"{sys._getframe(0).f_code.co_name} - Select {len(langs)} language.")

        for lang in langs:

            dict_path: str = os.path.join(dict_dir, f"{lang}.txt")
            is_file_dict_path: bool = os.path.isfile(dict_path)

            if is_file_dict_path:

                with open(dict_path, "r", encoding="utf-8") as f:

                    lang_dict: list = [l.rstrip() for l in f]

                self.dict.update({lang: lang_dict})

            else:

                # print(f"{sys._getframe(0).f_code.co_name} - Dictionary file is not exist. Filename : {dict_path}.")

                continue

    def recognize(self, image: np.ndarray, config: str="",
                #   language: str="kor", rank: int=0) -> str:
                  language: str="kor",  # Test.
                  rank: int=0) -> tuple[str, tuple[list, list]]:  # Test.

        langs: list = language.split("+")
        is_rank_under_langs: bool = rank < len(langs)

        if not is_rank_under_langs:

            # print(f"{sys._getframe(0).f_code.co_name} - Rank over than languages : {rank} < {len(langs)}. Set rank : 0.")

            rank: int = 0

        result: dict = {
            "confidence": [],
            "text": [],
        }

        for lang in langs:

            res: dict = self.__call__(image, lang, config)
            confs: list = res.get("conf", [-1])
            texts: list = res.get("text", [""])

            if confs:

                conf: int = -1  # Initialize.
                text: str = ""  # Initialize.
            
                indices: list = [i for i in range(len(confs)) if confs[i] >= 0]

                if indices:

                    confs: list = [confs[i] for i in indices]
                    conf: int = -int(sum(confs) / len(confs))  # Average.
                    texts: list = [texts[i] for i in indices]
                    text: str = "".join(texts)
                    flag: bool = self.__valid_text(lang, text)

                    if flag:

                        conf *= -1

                result["confidence"].append(conf)
                result["text"].append(text)

        confs: list = result.get("confidence")
        texts: list = result.get("text")

        confs_new: list = []
        texts_new: list = []

        for i, (conf, text) in enumerate(zip(confs, texts)):

            if rank == 0:

                confs_new.append(conf)
                texts_new.append(text)

            elif 0 < rank and i < rank and 0 <= conf:

                confs_new.append(conf)
                texts_new.append(text)

        if confs_new:

            text: str = texts_new[confs_new.index(max(confs_new))]

        else:
            
            text: str = texts[np.argsort(confs[rank:])[::-1].tolist()[0] + rank]

        # print(f"{sys._getframe(0).f_code.co_name} - confs : {confs}.")
        # print(f"{sys._getframe(0).f_code.co_name} - texts : {texts}.")

        # return text
        return text, (confs, texts)  # Test.

    def recognize_character(self, image: list, config: str="",
                            # language: str="kor", rank: int=0) -> str:
                            language: str="kor",  # Test.
                            rank: int=0) -> tuple[str, tuple[list, list]]:  # Test.

        temp: np.ndarray = image[0]

        if isinstance(temp, list):

            result: list = []  # Test.
            text: list = []

            while image:

                image_i: np.ndarray = image.pop(0)
                # txt: str = self.recognize_character(
                txt, res = self.recognize_character(  # Test.
                    image_i,
                    config,
                    language,
                    rank,
                )

                result.append(res)  # Test.
                text.append(txt)

        elif isinstance(temp, np.ndarray):

            result: list = []  # Test.
            text: list = []

            while image:

                image_i: np.ndarray = image.pop(0)
                # txt: str = self.recognize(
                txt, res = self.recognize(  # Test.
                    image_i,
                    config,
                    language,
                    rank,
                )

                result.append(res)  # Test.
                text.append(txt)

        else:

            ValueError("Image's type must be np.ndarray.")

        # return text
        return text, result  # Test.
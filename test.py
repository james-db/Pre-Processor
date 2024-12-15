DETECTION_MODEL_PATH: str = "D:/Code/003. Test/003. RAG/010. RAG Solution/test/Pre-processor/models/weights/craft_mlt_25k.pth"
# DETECTION_MODEL_PATH: str = "/home/jameskim/test/Pre-processor/models/weights/craft_mlt_25k.pth"
JSON_DIR: str = "D:/Code/003. Test/003. RAG/010. RAG Solution/sample/Advanced_Literate_Machinery_VGT_3"
# JSON_DIR: str = "/home/jameskim/test/sample/Advanced_Literate_Machinery_VGT_3"
PDF_DIR: str = "D:/Code/003. Test/003. RAG/010. RAG Solution/sample/data"
# PDF_DIR: str = "/home/jameskim/test/sample/data"
SAVE_DIR: str = "D:/Downloads/OCR"  # "D:/Code/003. Test/003. RAG/010. RAG Solution/sample/OCR"
# SAVE_DIR: str = "/home/jameskim/test/sample/OCR"

# Options
# Options : Pre & Post-Process.
BORDER: int = 10
DPI: int = 300
ITERATIONS: int = 3
JSON_DPI: int = 300

# Options : Character Detection.
CANVAS_SIZE: int = 1280
LOW_TEXT: float = 0.4
MAG_RATIO: float = 1.5

# Options : Character Recongnition - EasyOCR.
EASYOCR_LANGUAGE: str = ["ko","en"]

# Options : Character Recongnition - PaddleOCR.
PADDLE_LANGUAGE: str = "korean"

# Options : Character Recongnition - Surya.
SURYA_LANGUAGE: list = ["en", "ko"]

# Options : Character Recongnition - Tesseract.
TESSERACT_CMD: str = "C:/Users/DBInc/AppData/Local/Programs/Tesseract-OCR/tesseract"
# TESSERACT_CMD: str = "/usr/bin/tesseract"
TESSERACT_CONFIG: str = "--psm 10"
TESSERACT_LANGUAGE: str = "best/kor+kor+best/eng"

# Options : Document Analyzer.
TYPES: list = [
    "List-item",  # Advanced_Literate_Machinery_VGT.
    "Section-header",  # Advanced_Literate_Machinery_VGT.
    "Text",  # Advanced_Literate_Machinery_VGT, pdf-document-layout-analysis_VGT..
    "Title",  # Advanced_Literate_Machinery_VGT, pdf-document-layout-analysis_VGT.
]

# Options : Visualization.
FONT_PATH: str = "C:/Windows/Fonts/gulim.ttc"
# FONT_PATH: str = "/home/jameskim/test/Fonts/gulim.ttc"
FONT_SIZE: int = 40

import copy
import cv2
import datetime
import fitz  # Test : Will be removed.
from itertools import chain  # Test.
import json
import math
import natsort
import numpy as np
import os
# from pdf2image import convert_from_path  # Future : fitz -> pdf2image.
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)
import sys
import tqdm


from models.character_detector import Character_Detector
from models.character_recognizer import (
    EasyOCR,
    Paddle,
    Surya,
    Tesseract,
)
from utils.image import (
    crop_image,
    imshow,
    imwrite,
)
from utils.pdf import get_image
from utils.utils import (
    get_random_color,
    merge_coordinates,
    merge_text,
    scale_coordinates,
    shift_coordinates,
)


def recognize_character_easyocr(coordinates: tuple, image: np.ndarray,
                                border: int=0, dpi: int=72, ignore: int=5,
                                json_dpi: int=72, kernel_size: tuple=(3, 1),
                                space: int=5, text_threshold: float=0.4,
                                tolerance_factor: int=10,
                                visualize: bool=False) -> str:

    if isinstance(coordinates, list):

        coordinates: tuple = tuple(coordinates)

    coordinates: tuple = scale_coordinates(
        coordinates,
        dpi,
        json_dpi,
    )

    coordinates_backup: list = copy.deepcopy(coordinates)  # Copy for visualization.
    coordinates_0: tuple = copy.deepcopy(coordinates[:2])  # Copy for origin.

    # Pre-process for split per character.
    temp, _ = crop_image(
        coordinates,
        image,
    )

    # Split per character.
    coordinates, empty = character_detector.split_character(
        temp,
        ignore=ignore,
        kernel_size=kernel_size,
        space=space,
        text_threshold=text_threshold,
        tolerance_factor=tolerance_factor,
    )

    coordinates: list = shift_coordinates(
        coordinates,
        coordinates_0,
    )

    coordinates_backup: list = copy.deepcopy(coordinates)  # Copy for visualization.

    # print(f"{sys._getframe(0).f_code.co_name} - coordinates : {coordinates}.")  # Test.

    # Crop image per character.
    images, _ = crop_image(
        coordinates,
        # image,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        border,
        (255, 255, 255) if not empty else (0, 0, 0)
    )

    # OCR.
    text: str = recognizer_easy_ocr.recognize_character(
        images,
    )

    # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")  # Test.

    # Merge text row, space.
    text: str = merge_text(text)

    # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")  # Test.

    image_vis = None  # Test.

    # Visualization for test - Start.
    if visualize:

        image_vis_1: np.ndarray = image.copy()
        image_vis_2: np.ndarray = np.ones(
            (image_vis_1.shape[0], int(image_vis_1.shape[1] * 2), 3),
            np.uint8,
        ) * 255

        for j, coordinates_row in enumerate(coordinates_backup, 1):

            for k, coordinates_space in enumerate(coordinates_row, 1):

                top: bool = True  # Test.

                for l, (x_1, y_1, x_2, y_2) in enumerate(coordinates_space, 1):

                    # print(f"{sys._getframe(0).f_code.co_name} - (x_1, y_1, x_2, y_2) : {(x_1, y_1, x_2, y_2)}.")  # Test.

                    color: tuple = get_random_color()
                    cv2.rectangle(
                        image_vis_1,
                        (x_1, y_1),
                        (x_2, y_2),
                        color,
                        2,
                    )
                    cv2.putText(
                        image_vis_1,
                        f"{j}-{k}-{l}",
                        (x_1, y_1 - 5) if top else (x_1, y_2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    top: bool = not top

        image_vis_2 = Image.fromarray(image_vis_2)
        image_vis_2_draw = ImageDraw.Draw(image_vis_2)
        image_vis_2_draw.text(
            (0, 0),
            f"Text : \n{text}",
            (0, 0, 0),
            font=FONT,
        )

        image_vis: np.ndarray = cv2.hconcat(
            [
                cv2.cvtColor(image_vis_1, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(np.array(image_vis_2), cv2.COLOR_RGB2BGR),
            ],
        )
        # Visualization for test - End.

    return text, image_vis

def recognize_character_paddle(coordinates: tuple, image: np.ndarray,
                               border: int=0, classify: bool=False,
                               detect: bool=False, dpi: int=72, ignore: int=5,
                               json_dpi: int=72, kernel_size: tuple=(3, 1),
                               recognize: bool=True, space: int=5,
                               text_threshold: float=0.4,
                               tolerance_factor: int=10,
                               visualize: bool=False) -> str:

    if isinstance(coordinates, list):

        coordinates: tuple = tuple(coordinates)

    coordinates: tuple = scale_coordinates(
        coordinates,
        dpi,
        json_dpi,
    )

    coordinates_backup: list = copy.deepcopy(coordinates)  # Copy for visualization.
    coordinates_0: tuple = copy.deepcopy(coordinates[:2])  # Copy for origin.

    # Pre-process for split per character.
    temp, _ = crop_image(
        coordinates,
        image,
    )

    # Split per character.
    coordinates, empty = character_detector.split_character(
        temp,
        ignore=ignore,
        kernel_size=kernel_size,
        space=space,
        text_threshold=text_threshold,
        tolerance_factor=tolerance_factor,
    )

    coordinates: list = shift_coordinates(
        coordinates,
        coordinates_0,
    )

    coordinates_backup: list = copy.deepcopy(coordinates)  # Copy for visualization.

    # print(f"{sys._getframe(0).f_code.co_name} - coordinates : {coordinates}.")  # Test.

    # Crop image per character.
    images, _ = crop_image(
        coordinates,
        image,
        # cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        border,
        (255, 255, 255) if not empty else (0, 0, 0)
    )

    # OCR.
    text: str = recognizer_paddle.recognize_character(
        images,
        classify,
        detect,
        recognize,
    )

    # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")  # Test.

    # Merge text row, space.
    text: str = merge_text(text)

    print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")  # Test.

    image_vis = None  # Test.

    # Visualization for test - Start.
    if visualize:

        image_vis_1: np.ndarray = image.copy()
        image_vis_2: np.ndarray = np.ones(
            (image_vis_1.shape[0], int(image_vis_1.shape[1] * 2), 3),
            np.uint8,
        ) * 255

        for j, coordinates_row in enumerate(coordinates_backup, 1):

            for k, coordinates_space in enumerate(coordinates_row, 1):

                top: bool = True  # Test.

                for l, (x_1, y_1, x_2, y_2) in enumerate(coordinates_space, 1):

                    # print(f"{sys._getframe(0).f_code.co_name} - (x_1, y_1, x_2, y_2) : {(x_1, y_1, x_2, y_2)}.")  # Test.

                    color: tuple = get_random_color()
                    cv2.rectangle(
                        image_vis_1,
                        (x_1, y_1),
                        (x_2, y_2),
                        color,
                        2,
                    )
                    cv2.putText(
                        image_vis_1,
                        f"{j}-{k}-{l}",
                        (x_1, y_1 - 5) if top else (x_1, y_2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    top: bool = not top

        image_vis_2 = Image.fromarray(image_vis_2)
        image_vis_2_draw = ImageDraw.Draw(image_vis_2)
        image_vis_2_draw.text(
            (0, 0),
            f"Text : \n{text}",
            (0, 0, 0),
            font=FONT,
        )

        image_vis: np.ndarray = cv2.hconcat(
            [
                cv2.cvtColor(image_vis_1, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(np.array(image_vis_2), cv2.COLOR_RGB2BGR),
            ],
        )
        # Visualization for test - End.

    return text, image_vis

def recongnize_surya(coordinates: tuple, image: np.ndarray,
                     detect: bool=True, dpi: int=72, json_dpi: int=72,
                     language: list=["ko"], recognize: bool=True,
                    #  visualize: bool=False) -> str:
                     visualize: bool=False) -> tuple[str, np.ndarray]:  # Test.

    if isinstance(coordinates, list):

        coordinates: tuple = tuple(coordinates)

    coordinates: tuple = scale_coordinates(
        coordinates,
        dpi,
        json_dpi,
    )

    coordinates_backup: list = copy.deepcopy(coordinates)  # Copy for visualization.
    coordinates_0: tuple = copy.deepcopy(coordinates[:2])  # Copy for origin.

    temp, _ = crop_image(
        coordinates,
        image,
        0,
    )

    text: str = recognizer_surya.recognize(
        Image.fromarray(temp),
        detect,
        recognize,
    )

    # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")  # Test.

    coordinates: list = shift_coordinates(
        coordinates,
        coordinates_0,
    )

    image_vis = None  # Test.

    # Visualization for test - Start.
    if visualize:

        image_vis_1: np.ndarray = image.copy()
        image_vis_2: np.ndarray = np.ones(
            (image_vis_1.shape[0], int(image_vis_1.shape[1] * 2), 3),
            np.uint8,
        ) * 255

        image_vis_2 = Image.fromarray(image_vis_2)
        image_vis_2_draw = ImageDraw.Draw(image_vis_2)
        image_vis_2_draw.text(
            (0, 0),
            f"Text : \n{text}",
            (0, 0, 0),
            font=FONT,
        )

        image_vis: np.ndarray = cv2.hconcat(
            [
                cv2.cvtColor(image_vis_1, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(np.array(image_vis_2), cv2.COLOR_RGB2BGR),
            ],
        )
        # Visualization for test - End.

    # return text
    return text, image_vis  # Test.

def recognize_character_tesseract(coordinates: tuple, image: np.ndarray,
                                  border: int=0, config: str="",
                                  dpi: int=72, ignore: int=5,
                                  json_dpi: int=72, kernel_size: tuple=(3, 1),
                                  language: str="kor", rank: int=2,
                                  space: int=5, text_threshold: float=0.4,
                                  tolerance_factor: int=10,
                                #   visualize: bool=False) -> str:
                                  visualize: bool=False) -> tuple[str, list, np.ndarray]:  # Test.

    if isinstance(coordinates, list):

        coordinates: tuple = tuple(coordinates)

    coordinates: tuple = scale_coordinates(
        coordinates,
        dpi,
        json_dpi,
    )

    coordinates_backup: list = copy.deepcopy(coordinates)  # Copy for visualization.
    coordinates_0: tuple = copy.deepcopy(coordinates[:2])  # Copy for origin.

    # Pre-process for split per character.
    temp, _ = crop_image(
        coordinates,
        image,
    )

    # Split per character.
    coordinates, empty = character_detector.split_character(
        temp,
        ignore=ignore,
        kernel_size=kernel_size,
        space=space,
        text_threshold=text_threshold,
        tolerance_factor=tolerance_factor,
    )

    coordinates: list = shift_coordinates(
        coordinates,
        coordinates_0,
    )

    coordinates_backup: list = copy.deepcopy(coordinates)  # Copy for visualization.

    # print(f"{sys._getframe(0).f_code.co_name} - coordinates : {coordinates}.")  # Test.

    # Crop image per character.
    images, _ = crop_image(
        coordinates,
        image,
        border,
        (255, 255, 255) if not empty else (0, 0, 0)
    )

    # OCR.
    # text: str = recognizer_tesseract.recognize_character(
    text, result = recognizer_tesseract.recognize_character(  # Test.
        images,
        config,
        language,
        rank=rank,
    )

    # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")  # Test.

    # Merge text row, space.
    text: str = merge_text(text)
    result: list = list(chain.from_iterable(result))  # Test.

    # print(f"{sys._getframe(0).f_code.co_name} - text : {text}.")  # Test.
    # print(f"{sys._getframe(0).f_code.co_name} - result : {result}.")  # Test.

    result_new: list = []  # Test.

    for result_row in result:  # Test.

        for result_char in result_row:  # Test.

            for res in result_char:  # Test.

                result_new.append(str(res))  # Test.

    result: list = [  # Test.
        result_new[i:i + 30]
        for i in range(0, len(result_new), 30)
    ]
    result: list = [  # Test.
        "\n".join(result) for result in result
    ]

    image_vis = None  # Test.

    # Visualization for test - Start.
    if visualize:

        image_vis_1: np.ndarray = image.copy()
        image_vis_2: np.ndarray = np.ones(
            (image_vis_1.shape[0], int(image_vis_1.shape[1] * 2), 3),
            np.uint8,
        ) * 255

        for j, coordinates_row in enumerate(coordinates_backup, 1):

            for k, coordinates_space in enumerate(coordinates_row, 1):

                top: bool = True  # Test.

                for l, (x_1, y_1, x_2, y_2) in enumerate(coordinates_space, 1):

                    # print(f"{sys._getframe(0).f_code.co_name} - (x_1, y_1, x_2, y_2) : {(x_1, y_1, x_2, y_2)}.")  # Test.

                    color: tuple = get_random_color()
                    cv2.rectangle(
                        image_vis_1,
                        (x_1, y_1),
                        (x_2, y_2),
                        color,
                        2,
                    )
                    cv2.putText(
                        image_vis_1,
                        f"{j}-{k}-{l}",
                        (x_1, y_1 - 5) if top else (x_1, y_2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    top: bool = not top

        image_vis_2 = Image.fromarray(image_vis_2)
        image_vis_2_draw = ImageDraw.Draw(image_vis_2)
        image_vis_2_draw.text(
            (0, 0),
            f"Text : \n{text}",
            (0, 0, 0),
            font=FONT,
        )

        for j, res in enumerate(result):

            image_vis_2_draw.text(
                (int(image_vis_1.shape[1] / 3 * j), int(image_vis_1.shape[0] / 3)),
                f"Result : \n{res}",
                (0, 0, 0),
                font=FONT,
            )

        image_vis: np.ndarray = cv2.hconcat(
            [
                cv2.cvtColor(image_vis_1, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(np.array(image_vis_2), cv2.COLOR_RGB2BGR),
            ],
        )
        # Visualization for test - End.

    # return text
    return text, result, image_vis  # Test.

# Pillow.
FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)

if __name__ == "__main__":

    now: str = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")  # Test.
    save_dir: str = f"{SAVE_DIR}_{os.path.split(JSON_DIR)[1]}_{now}"  # Test.
    os.makedirs(save_dir, exist_ok=True)  # Test.

    ignore: int = int(math.ceil(DPI / 72) * 4)
    kernel_size_1: int = max(1, int(math.floor(DPI / 72 * 4)))
    kernel_size_2: int = max(1, int(math.floor(DPI / 72 * 1)))
    kernel_size: tuple = (kernel_size_1, kernel_size_2)
    space: int = int(math.ceil(DPI / 72) * 5)
    tolerance_factor: int = max(10, int(math.floor(DPI / 72 * 10)))

    fnames: list = natsort.natsorted(
        [
            fname
            for fname in os.listdir(PDF_DIR)
            if os.path.splitext(fname)[1].lower() == ".pdf"
        ],
    )

    character_detector: Character_Detector = Character_Detector()
    character_detector.build(DETECTION_MODEL_PATH)
    recognizer_easy_ocr: EasyOCR = EasyOCR(EASYOCR_LANGUAGE)
    recognizer_paddle: Paddle = Paddle(lang=PADDLE_LANGUAGE)
    recognizer_surya: Surya = Surya(SURYA_LANGUAGE)
    recognizer_tesseract: Tesseract = Tesseract(TESSERACT_CMD)
    recognizer_tesseract.add_dictionary(TESSERACT_LANGUAGE)

    for fname in tqdm.tqdm(fnames, desc="Document"):

        filename: str = os.path.splitext(fname)[0]
        document = fitz.open(os.path.join(PDF_DIR, fname))  # Test : Will be removed.
        # pages: list = convert_from_path(os.path.join(PDF_DIR, fname), DPI)  # Future : fitz -> pdf2image.

        json_fname: str = os.path.join(JSON_DIR, f"{filename}.json")
        is_file_json_fname: bool = os.path.isfile(json_fname)

        if is_file_json_fname:

            with open(json_fname, "r", encoding="utf-8") as f:

                layouts: list = json.load(f)

        else:

            print(f"Can not read file. Filename : {json_fname}")

            continue

        for i, layout in enumerate(tqdm.tqdm(layouts, desc="Layouts"), start=1):

            label: str = layout["label"]  # Advanced_Literate_Machinery_VGT.

            if label not in TYPES:

                continue

            coordinates: list = layout["bbox"]  # Advanced_Literate_Machinery_VGT.
            page_number: int = layout["page"]  # Advanced_Literate_Machinery_VGT, DocLayout-YOLO.
            image: np.ndarray = get_image(document, page_number, dpi=DPI)  # Test : Will be removed.
            # image: Image = pages[page_number - 1]  # Future : fitz -> pdf2image.

            # text, image_vis = recognize_character_easyocr(
            #     coordinates,
            #     image,
            #     BORDER,
            #     DPI,
            #     ignore,
            #     JSON_DPI,
            #     kernel_size,
            #     space,
            #     LOW_TEXT,
            #     tolerance_factor,
            #     True,
            # )
            text, image_vis = recognize_character_paddle(
                coordinates,
                image,
                BORDER,
                False,
                False,
                DPI,
                ignore,
                JSON_DPI,
                kernel_size,
                True,
                space,
                LOW_TEXT,
                tolerance_factor,
                True,
            )
            # text: str = recongnize_surya(
            # text, image_vis = recongnize_surya(  # Test.
            #     coordinates,
            #     image,
            #     True,
            #     DPI,
            #     JSON_DPI,
            #     SURYA_LANGUAGE,
            #     True,
            #     True,
            # )
            # text: str = recognize_character_tesseract(
            # text, _, image_vis = recognize_character_tesseract(  # Test.
            #     coordinates,
            #     image,
            #     BORDER,
            #     TESSERACT_CONFIG,
            #     DPI,
            #     ignore,
            #     JSON_DPI,
            #     kernel_size,
            #     TESSERACT_LANGUAGE,
            #     2,
            #     space,
            #     LOW_TEXT,
            #     tolerance_factor,
            #     True,
            # )

            imwrite(
                os.path.join(save_dir, f"{filename}_{page_number}_{i}.png"),
                image_vis,
            )
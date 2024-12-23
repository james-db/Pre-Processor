import copy
import numpy as np
import pdfplumber
import os
import pickle
import sys
from transformers import AutoTokenizer
import tqdm
import uuid


UUID: uuid.UUID = uuid.uuid1()
TEMP_DIR: str = os.path.join(
    os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],
    ".temp",
    str(UUID),
)

def create_grid_dict(page_data: list, tokenizer) -> dict:
    """Create a dictionary with the tokenized input, bounding box coordinates,
       and text.

    Args:
        page_data (list): List of word information from pdfplumber.
        tokenizer : HuggingFace Tokenizer.

    Returns:
        dict: Returns a dictionary with the tokenized input, bounding box
              coordinates, and text.
    """
    grid: dict = {
        "input_ids": [],
        "bbox_subword_list": [],
        "texts": [],
        "bbox_texts_list": []
    }

    if page_data:

        for ele in page_data:

            grid["texts"].append(ele["text"])

            # Since expected bbox format is (x, y, width, height).
            grid["bbox_texts_list"].append(
                (ele["x0"],
                ele["top"],
                ele["x1"]-ele["x0"],
                ele["bottom"]-ele["top"]))

        input_ids = tokenize(grid["texts"], tokenizer)

        # Flatten the input_ids.
        grid["bbox_subword_list"] = np.array(
            readjust_bbox_coords(
                grid["bbox_texts_list"],
                input_ids
            ),
        )
        grid["bbox_texts_list"] = np.array(grid["bbox_texts_list"])
        grid["input_ids"] = np.concatenate(input_ids)

    return grid

def organize(result: dict, thresh: float=0.5) -> dict:

    # print(f"{sys._getframe(0).f_code.co_name} - In : {result}.")

    categories: list = result.get("category")
    coordinates: list = result.get("coordinates")
    scores: list = result.get("score")

    if isinstance(categories, list):

        new_categoreis: np.ndarray = np.array(categories)

    elif isinstance(categories, np.ndarray):

        new_categoreis: np.ndarray = categories.copy()

    if isinstance(coordinates, list):

        new_coordinates: np.ndarray = np.array(coordinates)

    elif isinstance(coordinates, np.ndarray):

        new_coordinates: np.ndarray = coordinates.copy()

    if isinstance(scores, list):

        new_scores: np.ndarray = np.array(scores)

    elif isinstance(scores, np.ndarray):

        new_scores: np.ndarray = scores.copy()

    x_1: np.ndarray = new_coordinates[:, 0]
    x_2: np.ndarray = new_coordinates[:, 2]
    y_1: np.ndarray = new_coordinates[:, 1]
    y_2: np.ndarray = new_coordinates[:, 3]
    areas: np.ndarray = (x_2 - x_1) * (y_2 - y_1)
    order: np.ndarray = new_scores.argsort()[::-1]

    while order.size > 0:

        index: np.int64 = order[0]
        xx_1: np.ndarray = np.maximum(x_1[index], x_1[order[1:]])
        yy_1: np.ndarray = np.maximum(y_1[index], y_1[order[1:]])
        xx_2: np.ndarray = np.minimum(x_2[index], x_2[order[1:]])
        yy_2: np.ndarray = np.minimum(y_2[index], y_2[order[1:]])
        height: np.ndarray = np.maximum(0.0, yy_2 - yy_1)
        width: np.ndarray = np.maximum(0.0, xx_2 - xx_1)
        intersection: np.ndarray = width * height
        iou: np.ndarray = intersection / \
            (areas[index] + areas[order[1:]] - intersection)
        indices: np.ndarray = np.where(iou <= thresh)[0]
        overlap_indices: np.ndarray = order[np.where(iou > thresh)[0] + 1]

        if overlap_indices.size:

            overlap: np.ndarray = np.take(new_coordinates, overlap_indices, 0)

            # Set coordinates to uion.
            new_coordinates[index][0] = np.append(overlap[:, 0], x_1[index]).min()
            new_coordinates[index][1] = np.append(overlap[:, 1], y_1[index]).min()
            new_coordinates[index][2] = np.append(overlap[:, 2], x_2[index]).max()
            new_coordinates[index][3] = np.append(overlap[:, 3], y_2[index]).max()

            # Remove merged indices.
            new_categoreis: np.ndarray = np.delete(
                new_categoreis,
                overlap_indices,
                0,
            )
            new_coordinates: np.ndarray = np.delete(
                new_coordinates,
                overlap_indices,
                0,
            )
            new_scores: np.ndarray = np.delete(
                new_scores,
                overlap_indices,
                0,
            )

            # Reset.
            x_1: np.ndarray = new_coordinates[:, 0]
            x_2: np.ndarray = new_coordinates[:, 2]
            y_1: np.ndarray = new_coordinates[:, 1]
            y_2: np.ndarray = new_coordinates[:, 3]
            areas: np.ndarray = (x_2 - x_1) * (y_2 - y_1)
            order: np.ndarray = new_scores.argsort()[::-1]
            order: np.ndarray = order[np.where(order == index)[0][0]:]

        # To next index.
        else:

            order: np.ndarray = order[indices + 1]

    new_result: dict = {
        "category": new_categoreis.tolist(),
        "coordinates": new_coordinates.tolist(),
        "score": new_scores.tolist(),
    }

    # print(f"{sys._getframe(0).f_code.co_name} - Out : {new_result}.")

    return new_result

def pdf2image_and_pickle(fname: str, dpi: int=72,
                         tokenizer: str="google-bert/bert-base-uncased") -> str:

    print(f"{sys._getframe(0).f_code.co_name} - PDF converted to images with pickles and saved.")

    basename: str = os.path.basename(fname)
    image_dir: str = os.path.join(TEMP_DIR, basename, "image")
    pickle_dir: str = os.path.join(TEMP_DIR, basename, "pickle")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)

    tokenizer = select_tokenizer(tokenizer)

    document = pdfplumber.open(fname)

    for page in document.pages:

        page_number: int = page.page_number
        image = page.to_image(resolution=dpi)  # Resulution equals dpi.
        image.save(os.path.join(image_dir, f"page_{page_number}.png"))
        
        # Extracts words and their bounding boxes.
        words: list = page.extract_words()

        grid: dict = create_grid_dict(words, tokenizer)

        if isinstance(grid["input_ids"], list):
            
            print(f"{sys._getframe(0).f_code.co_name} - Page {page_number} has no wordgrid.")

        with open(
                os.path.join(pickle_dir,f"page_{page_number}.pkl"),
                "wb",
            ) as f:

            pickle.dump(grid, f)

    return image_dir, pickle_dir

def readjust_bbox_coords(bbox: list, tokens: list) -> list:
    """Readjust the bounding box coordinates based on the tokenized input.

    Args:
        bbox (list): List of bounding box coordinates in the format
                     (x, y, width, height).
        tokens (list): List of input_ids from the tokenizer.

    Returns:
        list: Returns a list of the adjusted bounding box coordinates.
    """
    adjusted_boxes = []

    for box, _id in zip(bbox, tokens):

        if len(_id) > 1:

            # Adjust the width and x-coordinate for each part.
            new_width = box[2] / len(_id)

            for i in range(len(_id)):

                adjusted_boxes.append(
                    (box[0] + i * new_width, box[1], new_width, box[3])
                )

        else:

            adjusted_boxes.append((box[0], box[1], box[2], box[3]))

    return adjusted_boxes

def select_tokenizer(model_name: str):
    """Select the tokenizer to be used.

    Args:
        tokenizer (str, optional): The name of the tokenizer to be used.
                                   Defaults to 
                                   "google-bert/bert-base-uncased".

    Returns:
        HuggingFace Tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer

def scale_coordinates2inch(coordinates, dpi: int=72) -> list:

    new_coordinates: list = list()

    for coor in coordinates:
        
        new_coor: tuple = tuple([c / dpi for c in coor])
        new_coordinates.append(new_coor)

    return new_coordinates

def tokenize(text_body: list, tokenizer) -> np.ndarray:
    """Tokenize the input text using the provided tokenizer.

    Args:
        text_body (list): List of text to be tokenized.
        tokenizer : HuggingFace Tokenizer.

    Returns:
        np.ndarray: Return the tokenized input_ids.
    """
    # Tokenize entire list of words.
    tokenized_inputs = tokenizer.batch_encode_plus(
        text_body,
        return_token_type_ids=False,
        return_attention_mask=False,
        add_special_tokens=False
    )
    return tokenized_inputs["input_ids"]
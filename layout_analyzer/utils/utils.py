import numpy as np
import pdfplumber
import pdf2image as _pdf2image
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

def get_word_grid(fname: str) -> list:
    """Return the word information from a PDF file using pdfplumber.

    Args:
        fname (str): The path to the input PDF file.

    Returns:
        list: Returns a list of shape (num_pages, num_words, 8).
    """
    doc = pdfplumber.open(fname)

    words = list()

    for page in doc.pages:
        
        # Extracts words and their bounding boxes.
        words.append(page.extract_words())

    return words

def pdf2image(fname: str, dpi: int=72) -> str:

    print(f"{sys._getframe(0).f_code.co_name} - PDF converted to images and saved.")

    # Convert the PDF to images.
    images = _pdf2image.convert_from_path(
        fname,
        dpi=dpi,  # Standard dpi used by pdfplumber is 72.
        fmt="png",
    )

    save_dir: str = os.path.join(TEMP_DIR, os.path.basename(fname), "image")
    os.makedirs(save_dir, exist_ok=True)

    # Save all images.
    for i, image in enumerate(tqdm.tqdm(images), 1):

        image.save(os.path.join(save_dir, f"page_{i}.png"))

    return save_dir

def pdf2pickle(fname: str,
               tokenizer: str="google-bert/bert-base-uncased") -> str:

    print(f"{sys._getframe(0).f_code.co_name} - PDF converted to pickles and saved.")

    tokenizer = select_tokenizer(tokenizer)
    wordgrid = get_word_grid(fname)

    save_dir: str = os.path.join(TEMP_DIR, os.path.basename(fname), "pickle")
    os.makedirs(save_dir, exist_ok=True)

    for i, page in enumerate(tqdm.tqdm(wordgrid), 1):

        page_data: list = wordgrid[i - 1]
        grid: dict = create_grid_dict(page_data, tokenizer)

        with open(os.path.join(save_dir, f"page_{i}.pkl"), "wb") as f:

            pickle.dump(grid, f)

    return save_dir

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
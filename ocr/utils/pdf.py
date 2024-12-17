import fitz
import numpy as np
from PIL import Image


# Test : Will be removed.
def get_image(document, page_number: int, dpi: int=72) -> np.ndarray:

    page = document.load_page(page_number - 1)
    pixmap = page.get_pixmap(dpi=dpi)
    image: Image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples,
    )
    image: np.ndarray = np.array(image)
    
    return image
from io import BytesIO

from PIL import Image
import fitz


def load_document_image(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc.load_page(0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

    return Image.open(BytesIO(file_bytes)).convert("RGB")

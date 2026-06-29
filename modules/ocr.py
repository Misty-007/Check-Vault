from dataclasses import dataclass
from pathlib import Path
import shutil

import fitz
import numpy as np
import pytesseract


WINDOWS_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


@dataclass
class OCRResult:
    text: str
    average_confidence: float
    source: str
    warning: str | None = None


def _extract_pdf_text(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text("text").strip() for page in doc]
    return "\n".join(page for page in pages if page)


def configure_tesseract():
    if shutil.which("tesseract"):
        return

    for path in WINDOWS_TESSERACT_PATHS:
        if path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            return


def extract_text(image, file_bytes=None, filename=""):
    configure_tesseract()

    if file_bytes and filename.lower().endswith(".pdf"):
        text = _extract_pdf_text(file_bytes)
        if text:
            return OCRResult(text=text, average_confidence=95.0, source="pdf_text")

    try:
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
    except Exception:
        return OCRResult(
            text="",
            average_confidence=0.0,
            source="unavailable",
            warning=(
                "Image OCR is unavailable because the Tesseract engine is not installed or not on PATH. "
                "PDF text extraction still works for digital PDFs."
            ),
        )

    words = []
    confidences = []

    for word, confidence in zip(data.get("text", []), data.get("conf", [])):
        word = word.strip()
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = -1

        if word:
            words.append(word)
        if confidence >= 0:
            confidences.append(confidence)

    average = float(np.mean(confidences)) if confidences else 0.0
    return OCRResult(text=" ".join(words), average_confidence=average, source="tesseract")

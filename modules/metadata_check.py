from dataclasses import dataclass
from io import BytesIO

import fitz
from PIL import Image


@dataclass
class MetadataResult:
    flags: list[str]
    details: dict[str, str]


def inspect_file(file_bytes, filename):
    flags = []
    details = {}
    lowered_filename = filename.lower()

    suspicious_keywords = [
        "edited",
        "fake",
        "modified",
        "photoshop",
        "copy",
        "final2",
        "new",
    ]

    for keyword in suspicious_keywords:
        if keyword in lowered_filename:
            flags.append(f"Suspicious filename keyword: {keyword}")

    if lowered_filename.endswith(".pdf"):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            metadata = doc.metadata or {}
            for key, value in metadata.items():
                if value:
                    details[key] = str(value)

            metadata_text = " ".join(details.values()).lower()
            if "photoshop" in metadata_text or "canva" in metadata_text:
                flags.append("PDF metadata references editing/design software")
            if metadata.get("creationDate") and metadata.get("modDate") and metadata["creationDate"] != metadata["modDate"]:
                flags.append("PDF modification date differs from creation date")
        except Exception:
            flags.append("Could not read PDF metadata")

    if lowered_filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            image = Image.open(BytesIO(file_bytes))
            details.update({str(key): str(value) for key, value in image.getexif().items()})
            size = len(file_bytes)
            if size < 20_000:
                flags.append("Very small file size for a document image")
        except Exception:
            pass

    return MetadataResult(flags=flags, details=details)

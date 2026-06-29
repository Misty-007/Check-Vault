from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.ela import run_ela
from modules.feature_extraction import extract_features
from modules.metadata_check import inspect_file
from modules.ocr import extract_text
from modules.pdf_utils import load_document_image


BASE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = BASE_DIR / "sample_documents"
OUTPUT_PATH = BASE_DIR / "training" / "training_features.csv"
SUPPORTED = {".jpg", ".jpeg", ".png", ".pdf"}


def analyze_file(path, label):
    file_bytes = path.read_bytes()
    image = load_document_image(file_bytes, path.name)
    metadata = inspect_file(file_bytes, path.name)
    ocr = extract_text(image, file_bytes=file_bytes, filename=path.name)
    ela = run_ela(image)
    features = extract_features(
        ocr_text=ocr.text,
        ocr_confidence=ocr.average_confidence,
        metadata_flags=metadata.flags,
        ela_score=ela.score,
        image=image,
        ela_image=ela.image,
        file_size_kb=len(file_bytes) / 1024,
    )
    return {"filename": path.name, "label": label, **features}


def main():
    rows = []
    for label in ["genuine", "tampered"]:
        folder = SAMPLE_DIR / label
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in SUPPORTED:
                rows.append(analyze_file(path, label))

    if not rows:
        raise FileNotFoundError(
            "No sample documents found. Run scripts/create_sample_docs.py first."
        )

    data = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(data)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

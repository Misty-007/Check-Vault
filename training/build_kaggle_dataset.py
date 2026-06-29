from pathlib import Path
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.ela import run_ela
from modules.feature_extraction import extract_features


BASE_DIR = Path(__file__).resolve().parents[1]
KAGGLE_DIR = (
    BASE_DIR
    / "external_data"
    / "tampered-documents-balanced-subset"
    / "dataset_copy_paste"
)
FALLBACK_KAGGLE_DIR = (
    BASE_DIR
    / "external_data"
    / "tampered-documents-subset"
    / "dataset_copy_paste"
)
SAMPLE_DIR = BASE_DIR / "sample_documents"
OUTPUT_PATH = BASE_DIR / "training" / "training_features.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_has_tamper(mask_path):
    import numpy as np

    mask = Image.open(mask_path).convert("L")
    return int(np.array(mask).sum()) > 0


def features_for_image(path, label, filename_prefix=""):
    file_bytes = path.read_bytes()
    image = Image.open(path).convert("RGB")
    ela = run_ela(image)
    features = extract_features(
        ocr_text="",
        ocr_confidence=0,
        metadata_flags=[],
        ela_score=ela.score,
        image=image,
        ela_image=ela.image,
        file_size_kb=len(file_bytes) / 1024,
    )
    return {"filename": f"{filename_prefix}{path.name}", "label": label, **features}


def kaggle_rows():
    rows = []
    kaggle_dir = KAGGLE_DIR if KAGGLE_DIR.exists() else FALLBACK_KAGGLE_DIR
    for split in ["train", "val", "test"]:
        image_dir = kaggle_dir / split / "images"
        mask_dir = kaggle_dir / split / "masks"
        if not image_dir.exists():
            continue

        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                continue
            label = "tampered" if image_has_tamper(mask_path) else "genuine"
            rows.append(features_for_image(image_path, label, f"kaggle_{split}_"))
    return rows


def sample_rows():
    rows = []
    for label in ["genuine", "tampered"]:
        folder = SAMPLE_DIR / label
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(features_for_image(path, label, "sample_"))
    return rows


def main():
    rows = kaggle_rows()
    label_counts = pd.Series([row["label"] for row in rows]).value_counts().to_dict()

    rows.extend(sample_rows())

    if len(label_counts) < 2:
        print(
            "Kaggle subset does not contain both classes, "
            "adding local sample images for balance."
        )

    if not rows:
        raise FileNotFoundError(
            "No training images found. Extract Kaggle subset or create sample docs first."
        )

    data = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(data)} rows to {OUTPUT_PATH}")
    print(data["label"].value_counts())


if __name__ == "__main__":
    main()

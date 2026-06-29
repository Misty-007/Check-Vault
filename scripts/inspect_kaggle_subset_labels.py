from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
SUBSET_DIR = BASE_DIR / "external_data" / "tampered-documents-subset" / "dataset_copy_paste"


def mask_has_tamper(mask_path):
    mask = Image.open(mask_path).convert("L")
    return int(np.array(mask).sum()) > 0


def main():
    if not SUBSET_DIR.exists():
        raise FileNotFoundError(
            f"{SUBSET_DIR} does not exist. Run scripts/extract_kaggle_subset.py first."
        )

    total = 0
    tampered = 0
    genuine = 0

    for split in ["train", "val", "test"]:
        mask_dir = SUBSET_DIR / split / "masks"
        masks = sorted(mask_dir.glob("*"))
        split_tampered = sum(1 for mask_path in masks if mask_has_tamper(mask_path))
        split_genuine = len(masks) - split_tampered

        total += len(masks)
        tampered += split_tampered
        genuine += split_genuine

        print(
            f"{split}: {len(masks)} masks, "
            f"{split_tampered} tampered, {split_genuine} genuine"
        )

    print(f"total: {total} masks, {tampered} tampered, {genuine} genuine")


if __name__ == "__main__":
    main()

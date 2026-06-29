from io import BytesIO
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
ZIP_PATH = BASE_DIR / "external_data" / "tampered-documents" / "tampered-documents.zip"
OUTPUT_DIR = BASE_DIR / "external_data" / "tampered-documents-balanced-subset"
SPLITS = ["train", "val", "test"]
PER_CLASS_LIMIT = 1200
RANDOM_SEED = 42


def mask_has_tamper(archive, mask_name):
    with archive.open(mask_name) as mask_file:
        mask = Image.open(BytesIO(mask_file.read())).convert("L")
    return int(np.array(mask).sum()) > 0


def image_pair_from_mask(mask_name, names):
    path = Path(mask_name)
    filename = path.name
    if not filename.endswith("_mask.png"):
        return None

    stem = filename[: -len("_mask.png")]
    split = path.parts[1]
    image_name = f"dataset_copy_paste/{split}/images/{stem}.png"
    label_name = f"dataset_copy_paste/{split}/labels/{stem}.txt"
    if image_name not in names:
        return None
    return image_name, mask_name, label_name if label_name in names else None


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"{ZIP_PATH} does not exist.")

    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())
        tampered = []
        genuine = []

        mask_names = [
            name
            for name in names
            if name.startswith("dataset_copy_paste/")
            and "/masks/" in name
            and name.endswith("_mask.png")
        ]
        mask_names.sort()

        for index, mask_name in enumerate(mask_names, start=1):
            pair = image_pair_from_mask(mask_name, names)
            if not pair:
                continue

            if mask_has_tamper(archive, mask_name):
                tampered.append(pair)
            else:
                genuine.append(pair)

            if index % 1000 == 0:
                print(
                    f"scanned {index}/{len(mask_names)} masks: "
                    f"{len(tampered)} tampered, {len(genuine)} genuine"
                )

        random.shuffle(tampered)
        random.shuffle(genuine)
        selected = tampered[:PER_CLASS_LIMIT] + genuine[:PER_CLASS_LIMIT]
        random.shuffle(selected)

        members = []
        for image_name, mask_name, label_name in selected:
            members.extend([image_name, mask_name])
            if label_name:
                members.append(label_name)

        for member in members:
            archive.extract(member, OUTPUT_DIR)

    print(f"Found {len(tampered)} tampered and {len(genuine)} genuine pairs")
    print(f"Extracted {len(selected)} balanced pairs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

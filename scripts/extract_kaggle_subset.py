from pathlib import Path
import zipfile


BASE_DIR = Path(__file__).resolve().parents[1]
ZIP_PATH = BASE_DIR / "external_data" / "tampered-documents" / "tampered-documents.zip"
OUTPUT_DIR = BASE_DIR / "external_data" / "tampered-documents-subset"
SPLITS = ["train", "val", "test"]
PER_SPLIT_LIMIT = 250


def paired_image_entries(names, split):
    prefix = f"dataset_copy_paste/{split}/images/"
    image_names = [name for name in names if name.startswith(prefix) and name.endswith(".png")]
    image_names.sort()
    pairs = []
    for image_name in image_names:
        filename = Path(image_name).name
        stem = filename[:-4]
        mask_name = f"dataset_copy_paste/{split}/masks/{stem}_mask.png"
        label_name = f"dataset_copy_paste/{split}/labels/{stem}.txt"
        if mask_name in names:
            pairs.append((image_name, mask_name, label_name if label_name in names else None))
        if len(pairs) >= PER_SPLIT_LIMIT:
            break
    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = set(archive.namelist())
        selected = []
        for split in SPLITS:
            pairs = paired_image_entries(names, split)
            for image_name, mask_name, label_name in pairs:
                selected.extend([image_name, mask_name])
                if label_name:
                    selected.append(label_name)
            print(f"{split}: selected {len(pairs)} image/mask pairs")

        for member in selected:
            archive.extract(member, OUTPUT_DIR)

    print(f"Extracted subset to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

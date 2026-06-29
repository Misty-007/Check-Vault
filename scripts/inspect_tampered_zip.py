from collections import Counter
from pathlib import Path
import zipfile


BASE_DIR = Path(__file__).resolve().parents[1]
ZIP_PATH = BASE_DIR / "external_data" / "tampered-documents" / "tampered-documents.zip"


def main():
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        counts = Counter()
        for name in names:
            parts = name.split("/")
            if len(parts) >= 4 and parts[0] == "dataset_copy_paste":
                counts["/".join(parts[:3])] += 1

    print(f"Zip: {ZIP_PATH}")
    print(f"Total files: {len(names)}")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

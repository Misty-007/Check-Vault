import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = BASE_DIR / "external_data" / "kaggle_kernel_source" / "text-tamper-detect-from-image.ipynb"
KEYWORDS = [
    "input",
    "dataset",
    "kaggle",
    "model",
    "train",
    "image",
    "tamper",
    "csv",
    ".h5",
    ".pt",
    ".pth",
    ".pkl",
]


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    print(f"Notebook: {NOTEBOOK}")
    print(f"Cells: {len(cells)}")

    for index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        lowered = source.lower()
        if any(keyword in lowered for keyword in KEYWORDS):
            print(f"\n--- cell {index} [{cell.get('cell_type')}] ---")
            print(source[:5000])


if __name__ == "__main__":
    main()

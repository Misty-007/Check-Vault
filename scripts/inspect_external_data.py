from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "external_data" / "text-tamper-detect-from-image"


def main():
    if not DATA_DIR.exists():
        print(f"Missing folder: {DATA_DIR}")
        print("Run .\\download_kaggle_output.ps1 first.")
        return

    files = [path for path in DATA_DIR.rglob("*") if path.is_file()]
    print(f"Found {len(files)} files in {DATA_DIR}")

    for path in files[:80]:
        size_kb = path.stat().st_size / 1024
        print(f"{path.relative_to(DATA_DIR)} ({size_kb:.1f} KB)")

    if len(files) > 80:
        print(f"... and {len(files) - 80} more files")


if __name__ == "__main__":
    main()

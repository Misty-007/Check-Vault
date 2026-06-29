from pathlib import Path
import shutil

import pytesseract


WINDOWS_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]


def main():
    binary = shutil.which("tesseract")
    if not binary:
        for path in WINDOWS_TESSERACT_PATHS:
            if path.exists():
                binary = str(path)
                pytesseract.pytesseract.tesseract_cmd = binary
                break

    if not binary:
        print("Tesseract engine: NOT FOUND")
        print("Install Tesseract OCR, then restart PowerShell.")
        print("Common Windows path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        return

    print(f"Tesseract engine: {binary}")
    print(f"pytesseract version: {pytesseract.get_tesseract_version()}")


if __name__ == "__main__":
    main()

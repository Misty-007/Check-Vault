from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = BASE_DIR / "sample_documents"


def _font(size):
    for name in ["arial.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_document(title, rows, output_path, tamper_box=None):
    width, height = 900, 1200
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((35, 35, width - 35, height - 35), outline=(30, 60, 100), width=3)
    draw.text((70, 70), title, fill=(20, 45, 90), font=_font(34))
    draw.line((70, 125, width - 70, 125), fill=(20, 45, 90), width=2)

    y = 175
    for label, value in rows:
        draw.text((90, y), label, fill=(30, 30, 30), font=_font(24))
        draw.text((440, y), value, fill=(30, 30, 30), font=_font(24))
        y += 58

    draw.text((90, 980), "Authorized Signatory", fill=(30, 30, 30), font=_font(22))
    draw.line((90, 1040, 360, 1040), fill=(30, 30, 30), width=2)
    draw.ellipse((615, 920, 790, 1095), outline=(140, 20, 20), width=5)
    draw.text((650, 985), "BANK", fill=(140, 20, 20), font=_font(28))

    if tamper_box:
        x1, y1, x2, y2, text = tamper_box
        draw.rectangle((x1, y1, x2, y2), fill=(250, 250, 250))
        draw.text((x1 + 8, y1 + 6), text, fill=(10, 10, 10), font=_font(24))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def _draw_pdf(title, rows, output_path, tamper_box=None):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    page.draw_rect(fitz.Rect(25, 25, 570, 817), color=(0.1, 0.2, 0.4), width=1.5)
    page.insert_text((55, 65), title, fontsize=22, color=(0.05, 0.15, 0.35))
    page.draw_line((55, 85), (540, 85), color=(0.05, 0.15, 0.35), width=1)

    y = 125
    for label, value in rows:
        page.insert_text((70, y), label, fontsize=13, color=(0.1, 0.1, 0.1))
        page.insert_text((300, y), value, fontsize=13, color=(0.1, 0.1, 0.1))
        y += 38

    page.insert_text((70, 700), "Authorized Signatory", fontsize=12, color=(0.1, 0.1, 0.1))
    page.draw_line((70, 735), (250, 735), color=(0.1, 0.1, 0.1), width=1)
    page.draw_oval(fitz.Rect(410, 660, 520, 770), color=(0.55, 0.08, 0.08), width=2)
    page.insert_text((432, 718), "BANK", fontsize=18, color=(0.55, 0.08, 0.08))

    if tamper_box:
        x1, y1, x2, y2, text = tamper_box
        page.draw_rect(fitz.Rect(x1, y1, x2, y2), color=(0.9, 0.9, 0.9), fill=(0.98, 0.98, 0.98))
        page.insert_text((x1 + 5, y1 + 19), text, fontsize=13, color=(0.02, 0.02, 0.02))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.set_metadata(
        {
            "title": title,
            "author": "CheckVault Demo",
            "creator": "CheckVault Sample Generator",
            "producer": "PyMuPDF",
        }
    )
    doc.save(output_path)
    doc.close()


def _create_document_pair(title, rows, output_path, tamper_box_image=None, tamper_box_pdf=None):
    _draw_document(title, rows, output_path.with_suffix(".jpg"), tamper_box=tamper_box_image)
    _draw_pdf(title, rows, output_path.with_suffix(".pdf"), tamper_box=tamper_box_pdf)


def main():
    genuine_rows = [
        ("Customer Name", "Aarav Sharma"),
        ("Account Number", "XXXX-4421"),
        ("Statement Period", "01/05/2026 - 31/05/2026"),
        ("Opening Balance", "INR 42,500"),
        ("Salary Credit", "INR 75,000"),
        ("EMI Debit", "INR 18,750"),
        ("Closing Balance", "INR 66,430"),
        ("Average Monthly Balance", "INR 51,300"),
        ("Document ID", "CV-BANK-2026-001"),
    ]

    tampered_rows = [
        ("Customer Name", "Aarav Sharma"),
        ("Account Number", "XXXX-4421"),
        ("Statement Period", "01/05/2026 - 31/05/2026"),
        ("Opening Balance", "INR 42,500"),
        ("Salary Credit", "INR 75,000"),
        ("EMI Debit", "INR 18,750"),
        ("Closing Balance", "INR 66,430"),
        ("Average Monthly Balance", "INR 51,300"),
        ("Document ID", "CV-BANK-2026-001"),
    ]

    _create_document_pair(
        "Clean Bank Statement",
        genuine_rows,
        SAMPLE_DIR / "genuine" / "clean_bank_statement_01",
    )
    _create_document_pair(
        "Clean Salary Slip",
        [
            ("Employee Name", "Nisha Rao"),
            ("Employee ID", "EMP-1427"),
            ("Pay Period", "May 2026"),
            ("Basic Salary", "INR 48,000"),
            ("Allowances", "INR 12,500"),
            ("Deductions", "INR 5,400"),
            ("Net Salary", "INR 55,100"),
            ("Employer", "Northline Systems Pvt Ltd"),
        ],
        SAMPLE_DIR / "genuine" / "clean_salary_slip_01",
    )
    _create_document_pair(
        "Modified Bank Statement",
        tampered_rows,
        SAMPLE_DIR / "tampered" / "edited_bank_statement_fake_01",
        tamper_box_image=(438, 518, 650, 558, "INR 6,66,430"),
        tamper_box_pdf=(298, 310, 430, 333, "INR 6,66,430"),
    )
    _create_document_pair(
        "Modified Salary Slip",
        [
            ("Employee Name", "Nisha Rao"),
            ("Employee ID", "EMP-1427"),
            ("Pay Period", "May 2026"),
            ("Basic Salary", "INR 48,000"),
            ("Allowances", "INR 12,500"),
            ("Deductions", "INR 5,400"),
            ("Net Salary", "INR 55,100"),
            ("Employer", "Northline Systems Pvt Ltd"),
        ],
        SAMPLE_DIR / "tampered" / "modified_salary_slip_fake_01",
        tamper_box_image=(438, 518, 650, 558, "INR 1,55,100"),
        tamper_box_pdf=(298, 310, 430, 333, "INR 1,55,100"),
    )

    print(f"Created sample documents in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()

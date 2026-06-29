# CheckVault

CheckVault is an offline AI-powered document forensics and underwriting intelligence prototype built for hackathon evaluation. It helps banking officers review uploaded financial and legal documents for signs of tampering, suspicious metadata, abnormal OCR output, and high-risk underwriting signals.

## One-Line Pitch

CheckVault detects forged or manipulated financial documents using OCR, ELA heatmaps, metadata inspection, and a locally trained fraud-risk model to support faster and safer underwriting decisions.

## Problem

Banks and financial institutions manually review large volumes of salary slips, bank statements, land records, and legal documents during loan approvals, KYC, and property validation. Fraudsters can digitally edit numbers, paste signatures, manipulate seals, or alter scanned documents, making manual verification slow and unreliable.

## Solution

CheckVault provides a hybrid verification pipeline:

1. Upload a PDF or image document.
2. Extract text with OCR or embedded PDF text extraction.
3. Inspect metadata and suspicious filename/document signals.
4. Generate an Error Level Analysis heatmap.
5. Extract visual forensic features from the document.
6. Run a local machine learning model.
7. Produce a fraud risk score and underwriting recommendation.

The system runs offline after setup. No external API key is required during the demo.

## Core Features

- PDF and image upload
- OCR-based text extraction
- Manual text fallback if OCR is unavailable
- Error Level Analysis heatmap
- Metadata and filename risk checks
- Local trained ML model
- Fraud risk score from 0 to 100
- Recommendation: Approve, Manual Review, or Reject / Investigate
- Downloadable feature CSV for explainability

## Model Summary

The current local classifier uses:

- ELA statistics
- image edge density
- image brightness/darkness ratios
- image entropy
- file size
- OCR confidence and text length
- metadata flag count
- extracted financial amount patterns

Training uses a balanced subset from a document tampering dataset plus local sample documents. The latest validation result is approximately:

```text
Macro F1: 0.752
Accuracy: 0.75
```

This prototype is designed as a decision-support and triage layer, not as a final autonomous banking approval system.

## Tech Stack

- Python
- Streamlit
- OpenCV/Pillow
- PyMuPDF
- pytesseract
- scikit-learn
- pandas

## Run Locally

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

Or on Windows, use:

```powershell
.\run_app.ps1
```

Open:

```text
http://localhost:8501
```

## OCR Setup

For image OCR, install the Tesseract desktop engine.

Recommended Windows path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Check OCR setup:

```powershell
python scripts\check_ocr_setup.py
```

If Tesseract is not installed, the app still runs. PDF embedded text extraction, ELA heatmaps, metadata checks, model prediction, and risk scoring remain available.

## Train The Model

To train with the current feature CSV:

```powershell
python training\train_model.py
```

To create sample documents and train a small demo model:

```powershell
.\train_demo_model.ps1
```

To train from the Kaggle tampered-document dataset, download the dataset zip to:

```text
external_data/tampered-documents/tampered-documents.zip
```

Then run:

```powershell
python scripts\extract_kaggle_balanced_subset.py
python training\build_kaggle_dataset.py
python training\train_model.py
```

The trained model is saved at:

```text
models/checkvault_model.pkl
```

## Repository Notes

The repository should include:

```text
app.py
modules/
training/
scripts/
models/checkvault_model.pkl
models/model_report.txt
models/feature_importance.csv
sample_documents/
requirements.txt
run_app.ps1
README.md
```

Do not upload:

```text
external_data/
.kaggle/
venv/
__pycache__/
```

## Demo Flow For Judges

1. Start the Streamlit app.
2. Upload one sample document from `sample_documents/`.
3. Show the original document and ELA heatmap.
4. Point out OCR, metadata, and feature highlights.
5. Explain the model signal and fraud risk score.
6. End with the underwriting recommendation.

## Limitations And Future Work

- Current model is a lightweight forensic classifier, not a full deep-learning tamper localization model.
- OCR quality depends on document scan quality and Tesseract installation.
- More real bank statements, salary slips, land records, and legal documents would improve domain accuracy.
- Future versions can add CNN/ResNet localization, signature verification, seal detection, and document-type-specific models.

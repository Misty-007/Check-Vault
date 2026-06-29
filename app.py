from pathlib import Path

import pandas as pd
import streamlit as st

from modules.ela import run_ela
from modules.feature_extraction import extract_features
from modules.metadata_check import inspect_file
from modules.ocr import extract_text
from modules.pdf_utils import load_document_image
from modules.risk_engine import calculate_risk
from modules.trained_model import predict_with_model


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "checkvault_model.pkl"


st.set_page_config(page_title="CheckVault", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --cv-pink: #e83e8c;
        --cv-pink-dark: #a9145f;
        --cv-pink-soft: #fff0f7;
        --cv-white: #ffffff;
        --cv-ink: #211827;
        --cv-muted: #786070;
        --cv-border: #f4b6d2;
        --cv-card: rgba(255, 255, 255, 0.92);
        --cv-shadow: 0 18px 45px rgba(169, 20, 95, 0.11);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(232, 62, 140, 0.14), transparent 34rem),
            radial-gradient(circle at bottom right, rgba(244, 182, 210, 0.20), transparent 30rem),
            linear-gradient(135deg, #fff9fc 0%, #ffffff 46%, #fff4fa 100%);
        color: var(--cv-ink);
    }

    .block-container {
        width: min(1500px, calc(100vw - 4rem)) !important;
        max-width: 1500px !important;
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    div[data-testid="stMainBlockContainer"] {
        width: min(1500px, calc(100vw - 4rem)) !important;
        max-width: 1500px !important;
    }

    .cv-hero {
        position: relative;
        overflow: hidden;
        padding: 1.8rem 2rem;
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(33, 24, 39, 0.88), rgba(169, 20, 95, 0.84)),
            linear-gradient(135deg, #e83e8c 0%, #f7a6cc 100%);
        box-shadow: 0 28px 70px rgba(169, 20, 95, 0.22);
        color: white;
        margin-bottom: 1rem;
    }

    .cv-hero:after {
        content: "";
        position: absolute;
        inset: auto -5rem -8rem auto;
        width: 24rem;
        height: 24rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15);
    }

    .cv-kicker {
        font-size: 0.82rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 800;
        opacity: 0.92;
        margin-bottom: 0.5rem;
    }

    .cv-title {
    
        font-size: clamp(2.4rem, 4.2vw, 4.2rem);
        line-height: 0.95;
        font-weight: 900;
        margin: 0;
    }

    .cv-subtitle {
        max-width: 980px;
        font-size: 1.08rem;
        opacity: 0.96;
        margin-top: 1rem;
        margin-bottom: 0;
    }

    .cv-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 1.3rem;
    }

    .cv-chip {
        border: 1px solid rgba(255, 255, 255, 0.55);
        background: rgba(255, 255, 255, 0.18);
        color: white;
        border-radius: 999px;
        padding: 0.45rem 0.85rem;
        font-weight: 700;
        backdrop-filter: blur(6px);
    }

    .cv-panel {
        background: var(--cv-card);
        border: 1px solid #f8c7dc;
        border-radius: 24px;
        padding: 1.25rem;
        box-shadow: var(--cv-shadow);
        margin-bottom: 1rem;
    }

    .cv-disclaimer {
        background: #fff7fb;
        border: 1px solid #f8c7dc;
        border-left: 6px solid var(--cv-pink);
        border-radius: 18px;
        color: var(--cv-muted);
        padding: 0.9rem 1rem;
        margin: 0.75rem 0 1rem;
        font-size: 0.94rem;
    }

    .cv-small-card {
        background: var(--cv-card);
        border: 1px solid #f8c7dc;
        border-radius: 20px;
        padding: 1rem;
        min-height: 104px;
        box-shadow: var(--cv-shadow);
    }

    .cv-small-card-label {
        color: var(--cv-muted);
        font-size: 0.78rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }

    .cv-small-card-value {
        color: var(--cv-pink-dark);
        font-size: 1.45rem;
        font-weight: 950;
        line-height: 1.05;
    }

    .cv-small-card-note {
        color: var(--cv-muted);
        font-size: 0.84rem;
        margin-top: 0.35rem;
    }

    .cv-panel-title {
        color: var(--cv-pink-dark);
        font-size: 0.86rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .cv-panel-text {
        color: var(--cv-muted);
        margin: 0;
    }

    .cv-section {
        color: var(--cv-ink);
        font-size: 1.35rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        margin: 1.3rem 0 0.65rem;
    }

    div[data-testid="stFileUploader"] {
        background: var(--cv-card);
        border: 2px dashed var(--cv-border);
        border-radius: 22px;
        padding: 1rem;
        box-shadow: var(--cv-shadow);
    }

    div[data-testid="stFileUploader"] label,
    .stMarkdown, .stTextArea label {
        color: var(--cv-ink);
    }

    div[data-testid="stMetric"] {
        background: var(--cv-card);
        border: 1px solid #fbcfe8;
        border-radius: 22px;
        padding: 1rem;
        box-shadow: var(--cv-shadow);
    }

    div[data-testid="stMetric"] label {
        color: var(--cv-muted);
        font-weight: 800;
    }

    div[data-testid="stMetricValue"] {
        color: var(--cv-pink-dark);
        font-weight: 900;
    }

    h2, h3 {
        color: var(--cv-pink-dark);
        letter-spacing: -0.02em;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--cv-pink), var(--cv-pink-dark));
        color: white;
        border: 0;
        border-radius: 999px;
        padding: 0.65rem 1.1rem;
        font-weight: 800;
        box-shadow: 0 12px 28px rgba(190, 24, 93, 0.24);
    }

    .stTextArea textarea,
    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        border-color: #fbcfe8;
    }

    div[data-testid="stImage"] img {
        border-radius: 20px;
        border: 1px solid #f8c7dc;
        box-shadow: var(--cv-shadow);
        max-height: 620px;
        object-fit: contain;
    }

    .stAlert {
        border-radius: 18px;
    }

    hr {
        border-color: #fbcfe8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cv-hero">
        <div class="cv-kicker">Underwriting Intelligence Suite</div>
        <h1 class="cv-title">CheckVault</h1>
        <p class="cv-subtitle">
            Offline document forensics for loan, KYC, salary-slip, and property-file
            verification with explainable forgery signals and risk recommendations.
        </p>
        <div class="cv-chip-row">
            <span class="cv-chip">Offline OCR</span>
            <span class="cv-chip">ELA Heatmaps</span>
            <span class="cv-chip">Forgery Signals</span>
            <span class="cv-chip">Risk Recommendation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cv-panel">
        <div class="cv-panel-title">Verification Workflow</div>
        <p class="cv-panel-text">
            Upload a document, review OCR and metadata signals, inspect the ELA heatmap,
            then use the local model score as an underwriting decision aid.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cv-disclaimer">
        Prototype note: CheckVault is a hackathon-stage decision-support tool.
        It explains document risk signals for reviewers, but final approval should
        remain with a human banking officer.
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a document for CheckVault analysis",
    type=["png", "jpg", "jpeg", "pdf"],
)

if not uploaded_file:
    st.info("Upload a PDF or image to begin analysis. Try the generated samples in `sample_documents` after running the sample script.")
    st.stop()

file_bytes = uploaded_file.getvalue()
image = load_document_image(file_bytes, uploaded_file.name)
metadata_result = inspect_file(file_bytes, uploaded_file.name)
ocr_result = extract_text(image, file_bytes=file_bytes, filename=uploaded_file.name)
ela_result = run_ela(image)

manual_text = ""
if ocr_result.warning and not ocr_result.text:
    st.warning(ocr_result.warning)
    manual_text = st.text_area(
        "Optional fallback: paste document text here for financial/risk analysis",
        height=140,
        placeholder="Example: Net Salary INR 55,100, Closing Balance INR 66,430...",
    )

analysis_text = manual_text.strip() or ocr_result.text

features = extract_features(
    ocr_text=analysis_text,
    ocr_confidence=ocr_result.average_confidence,
    metadata_flags=metadata_result.flags,
    ela_score=ela_result.score,
    image=image,
    ela_image=ela_result.image,
    file_size_kb=len(file_bytes) / 1024,
)

model_result = predict_with_model(MODEL_PATH, features)
risk_result = calculate_risk(features, metadata_result.flags, model_result)

feature_df = pd.DataFrame([features])
text_source = ocr_result.source if ocr_result.text else "manual fallback" if manual_text else "none"

summary_cols = st.columns(4)
with summary_cols[0]:
    st.markdown(
        f"""
        <div class="cv-small-card">
            <div class="cv-small-card-label">Document</div>
            <div class="cv-small-card-value">{uploaded_file.name[:28]}</div>
            <div class="cv-small-card-note">{len(file_bytes) / 1024:.1f} KB uploaded</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with summary_cols[1]:
    st.markdown(
        f"""
        <div class="cv-small-card">
            <div class="cv-small-card-label">OCR Status</div>
            <div class="cv-small-card-value">{text_source}</div>
            <div class="cv-small-card-note">{ocr_result.average_confidence:.1f}% confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with summary_cols[2]:
    st.markdown(
        f"""
        <div class="cv-small-card">
            <div class="cv-small-card-label">Risk Decision</div>
            <div class="cv-small-card-value">{risk_result.recommendation}</div>
            <div class="cv-small-card-note">Fraud score {risk_result.score}/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with summary_cols[3]:
    st.markdown(
        f"""
        <div class="cv-small-card">
            <div class="cv-small-card-label">ML Signal</div>
            <div class="cv-small-card-value">{model_result.label}</div>
            <div class="cv-small-card-note">{model_result.confidence:.2f} confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="cv-section">Document Forensics</div>', unsafe_allow_html=True)

left, right, side = st.columns([1.2, 1.2, 0.8])

with left:
    st.subheader("Original Document")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Tamper Heatmap")
    st.image(ela_result.image, use_container_width=True)

with side:
    st.subheader("Forensic Signals")
    st.metric("ELA Score", ela_result.score)
    st.metric("OCR Confidence", f"{ocr_result.average_confidence:.1f}%")
    st.metric("Metadata Flags", len(metadata_result.flags))
    st.metric("Text Length", len(analysis_text or ""))

st.divider()

st.markdown('<div class="cv-section">Underwriting Decision</div>', unsafe_allow_html=True)

score_col, decision_col, model_col = st.columns(3)
score_col.metric("Fraud Risk Score", f"{risk_result.score}/100")
decision_col.metric("Recommendation", risk_result.recommendation)
model_col.metric("Model Signal", model_result.label)

st.subheader("Decision Rationale")
for reason in risk_result.reasons:
    st.write(f"- {reason}")

detail_left, detail_right = st.columns([1.15, 0.85])

with detail_left:
    st.subheader("Extracted Text")
    st.caption(f"Text source: {text_source}")
    st.text_area("Text Used For Analysis", analysis_text or "No text extracted.", height=240)

with detail_right:
    st.subheader("Metadata Review")
    if metadata_result.flags:
        for flag in metadata_result.flags:
            st.warning(flag)
    else:
        st.success("No suspicious metadata signals found.")

    st.subheader("Feature Highlights")
    highlight_df = pd.DataFrame(
        [
            {"feature": "ela_mean", "value": round(features.get("ela_mean", 0), 3)},
            {"feature": "ela_high_ratio", "value": round(features.get("ela_high_ratio", 0), 3)},
            {"feature": "image_edge_density", "value": round(features.get("image_edge_density", 0), 3)},
            {"feature": "image_entropy", "value": round(features.get("image_entropy", 0), 3)},
            {"feature": "file_size_kb", "value": round(features.get("file_size_kb", 0), 3)},
        ]
    )
    st.dataframe(highlight_df, use_container_width=True, hide_index=True)

st.subheader("Complete Model Feature Vector")
st.dataframe(feature_df, use_container_width=True, hide_index=True)

st.download_button(
    "Download Feature CSV",
    data=feature_df.assign(filename=uploaded_file.name).to_csv(index=False),
    file_name="checkvault_features.csv",
    mime="text/csv",
)

st.markdown(
    """
    <div class="cv-disclaimer">
        Model transparency: the local classifier was trained on a balanced tampered-document
        subset using ELA, image statistics, OCR-derived signals, and metadata features.
        Current validation macro F1 is approximately 0.75, so the output is best used
        as a triage and explainability layer rather than an autonomous rejection system.
    </div>
    """,
    unsafe_allow_html=True,
)

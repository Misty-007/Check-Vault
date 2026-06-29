from dataclasses import dataclass


@dataclass
class RiskResult:
    score: int
    recommendation: str
    reasons: list[str]


def calculate_risk(features, metadata_flags, model_result):
    score = 0
    reasons = []

    if features["ela_score"] >= 80:
        score += 35
        reasons.append("High ELA inconsistency detected")
    elif features["ela_score"] >= 45:
        score += 20
        reasons.append("Moderate ELA inconsistency detected")

    if metadata_flags:
        score += min(25, len(metadata_flags) * 10)
        reasons.append("Suspicious metadata or filename signals detected")

    if features["ocr_text_length"] < 50:
        score += 15
        reasons.append("Very low OCR text extraction")

    if features["ocr_confidence"] and features["ocr_confidence"] < 45:
        score += 10
        reasons.append("Low OCR confidence")

    if features["amount_variance"] > 1_000_000_000:
        score += 15
        reasons.append("Large financial amount variation detected")

    if features["suspicious_keyword_count"] > 0:
        score += 15
        reasons.append("Suspicious keywords found in extracted text")

    if model_result.available:
        model_label = model_result.label.lower()
        if model_label in {"tampered", "fraud"}:
            score += 50
            reasons.append(f"Local ML model predicted: {model_result.label}")
        elif model_label == "suspicious":
            score += 30
            reasons.append(f"Local ML model predicted: {model_result.label}")
        else:
            reasons.append(f"Local ML model predicted: {model_result.label}")

    score = min(score, 100)

    if score <= 30:
        recommendation = "Approve"
    elif score <= 65:
        recommendation = "Manual Review"
    else:
        recommendation = "Reject / Investigate"

    if not reasons:
        reasons.append("No major suspicious signals detected")

    return RiskResult(score=score, recommendation=recommendation, reasons=reasons)

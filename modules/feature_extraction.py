import re

import numpy as np
from PIL import ImageFilter, ImageStat


AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹)?\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?", re.I)
DATE_PATTERN = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")


def _extract_amounts(text):
    amounts = []
    for match in AMOUNT_PATTERN.findall(text or ""):
        value = match.replace(",", "")
        try:
            amounts.append(float(value))
        except ValueError:
            continue
    return amounts


def _image_features(image):
    if image is None:
        return {
            "image_width": 0.0,
            "image_height": 0.0,
            "image_aspect_ratio": 0.0,
            "image_mean": 0.0,
            "image_std": 0.0,
            "image_entropy": 0.0,
            "image_dark_ratio": 0.0,
            "image_light_ratio": 0.0,
            "image_edge_density": 0.0,
            "red_mean": 0.0,
            "green_mean": 0.0,
            "blue_mean": 0.0,
            "red_std": 0.0,
            "green_std": 0.0,
            "blue_std": 0.0,
        }

    rgb = image.convert("RGB")
    grayscale = image.convert("L")
    array = np.array(grayscale)
    edges = np.array(grayscale.filter(ImageFilter.FIND_EDGES))
    color_stat = ImageStat.Stat(rgb)
    return {
        "image_width": float(image.width),
        "image_height": float(image.height),
        "image_aspect_ratio": float(image.width / image.height if image.height else 0),
        "image_mean": float(np.mean(array)),
        "image_std": float(np.std(array)),
        "image_entropy": float(grayscale.entropy()),
        "image_dark_ratio": float(np.mean(array < 40)),
        "image_light_ratio": float(np.mean(array > 215)),
        "image_edge_density": float(np.mean(edges > 30)),
        "red_mean": float(color_stat.mean[0]),
        "green_mean": float(color_stat.mean[1]),
        "blue_mean": float(color_stat.mean[2]),
        "red_std": float(color_stat.stddev[0]),
        "green_std": float(color_stat.stddev[1]),
        "blue_std": float(color_stat.stddev[2]),
    }


def _ela_features(ela_image, ela_score):
    if ela_image is None:
        return {
            "ela_score": float(ela_score),
            "ela_mean": 0.0,
            "ela_std": 0.0,
            "ela_p95": float(ela_score),
            "ela_p99": float(ela_score),
            "ela_max": float(ela_score),
            "ela_high_ratio": 0.0,
        }

    ela_array = np.array(ela_image.convert("L"))
    return {
        "ela_score": float(ela_score),
        "ela_mean": float(np.mean(ela_array)),
        "ela_std": float(np.std(ela_array)),
        "ela_p95": float(np.percentile(ela_array, 95)),
        "ela_p99": float(np.percentile(ela_array, 99)),
        "ela_max": float(np.max(ela_array)),
        "ela_high_ratio": float(np.mean(ela_array > 40)),
    }


def extract_features(
    ocr_text,
    ocr_confidence,
    metadata_flags,
    ela_score,
    image=None,
    ela_image=None,
    file_size_kb=0,
):
    amounts = _extract_amounts(ocr_text)
    dates = DATE_PATTERN.findall(ocr_text or "")
    lowered = (ocr_text or "").lower()
    suspicious_words = ["overwritten", "duplicate", "invalid", "cancelled", "mismatch", "fake"]
    suspicious_keyword_count = sum(1 for word in suspicious_words if word in lowered)

    features = {
        "file_size_kb": float(file_size_kb),
        "metadata_suspicious_count": float(len(metadata_flags)),
        "ocr_text_length": float(len(ocr_text or "")),
        "ocr_confidence": float(ocr_confidence),
        "amount_count": float(len(amounts)),
        "max_amount": float(max(amounts) if amounts else 0),
        "average_amount": float(np.mean(amounts) if amounts else 0),
        "amount_variance": float(np.var(amounts) if len(amounts) > 1 else 0),
        "date_count": float(len(dates)),
        "suspicious_keyword_count": float(suspicious_keyword_count),
    }
    features.update(_ela_features(ela_image, ela_score))
    features.update(_image_features(image))
    return features

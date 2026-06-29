from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class ModelResult:
    label: str
    confidence: float
    available: bool


def predict_with_model(model_path, features):
    model_path = Path(model_path)
    if not model_path.exists():
        return ModelResult(label="No trained model yet", confidence=0.0, available=False)

    model = joblib.load(model_path)
    frame = pd.DataFrame([features])
    prediction = str(model.predict(frame)[0])

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(frame).max())

    return ModelResult(label=prediction, confidence=confidence, available=True)

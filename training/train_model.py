from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "training" / "training_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "checkvault_model.pkl"
REPORT_PATH = MODEL_DIR / "model_report.txt"
IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Create training/training_features.csv first. "
            "It must include the feature columns plus a label column."
        )

    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=["filename", "label"], errors="ignore")
    y = data["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results = []
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        predictions = candidate.predict(x_test)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        results.append((macro_f1, name, candidate, predictions))
        print(f"\n{name} macro f1: {macro_f1:.3f}")
        print(classification_report(y_test, predictions))

    best_f1, best_name, model, predictions = max(results, key=lambda item: item[0])
    report = (
        f"Best model: {best_name}\n"
        f"Macro F1: {best_f1:.3f}\n\n"
        f"{classification_report(y_test, predictions)}"
    )
    print(report)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved report to {REPORT_PATH}")

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": x.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(IMPORTANCE_PATH, index=False)
        print(f"Saved feature importance to {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()

"""
root_cause_classifier.py
=========================
Model 3 for TextileGuard AI: predicts the ROOT CAUSE of a detected fabric
defect (machine_fault / material_issue / human_error) from production-line
context, given as a CSV row or JSON payload.

This module is independent of preprocessing.py (YOLOv8) and stain_classifier.py
(MobileNetV2) — it operates on tabular/CSV data, not images, so it has no
image preprocessing needs at all.

Usage (mirrors StainClassifier's load-once-at-startup pattern):

    from root_cause_classifier import RootCauseClassifier
    root_cause_model = RootCauseClassifier("root_cause_model.joblib")
    result = root_cause_model.predict(single_row_dict)
    results = root_cause_model.predict_batch(dataframe)
"""

from typing import Dict, List

import joblib
import pandas as pd


class RootCauseClassifier:
    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]                # sklearn Pipeline (preprocess + XGBoost)
        self.label_encoder = bundle["label_encoder"]
        self.categorical = bundle["categorical"]
        self.numeric = bundle["numeric"]
        self.required_columns = self.categorical + self.numeric

    def _validate(self, df: pd.DataFrame) -> List[str]:
        missing = [c for c in self.required_columns if c not in df.columns]
        return missing

    def predict(self, row: Dict) -> Dict:
        """Predict root cause for a single defect record (dict of fields)."""
        df = pd.DataFrame([row])
        missing = self._validate(df)
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        df = df[self.required_columns]
        proba = self.model.predict_proba(df)[0]
        pred_idx = proba.argmax()
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        return {
            "predicted_cause": pred_label,
            "confidence": float(proba[pred_idx]),
            "probabilities": {
                cls: float(p) for cls, p in zip(self.label_encoder.classes_, proba)
            },
        }

    def predict_batch(self, df: pd.DataFrame) -> List[Dict]:
        """Predict root cause for every row in an uploaded CSV (as a DataFrame)."""
        missing = self._validate(df)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        X = df[self.required_columns]
        proba = self.model.predict_proba(X)
        pred_idx = proba.argmax(axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_idx)

        results = []
        for i in range(len(df)):
            results.append({
                "row": i,
                "predicted_cause": pred_labels[i],
                "confidence": float(proba[i, pred_idx[i]]),
                "probabilities": {
                    cls: float(p) for cls, p in zip(self.label_encoder.classes_, proba[i])
                },
            })
        return results

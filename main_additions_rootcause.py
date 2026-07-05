# =============================================================
# ADD TO main.py — Model 3: Root Cause Predictor
# =============================================================
# 1) Add this import near your other model imports:
#
#    from root_cause_classifier import RootCauseClassifier
#
# 2) Add this to your startup event, alongside the YOLOv8 and
#    stain model loading:
#
#    root_cause_model = RootCauseClassifier("root_cause_model.joblib")
#
# 3) Paste these two endpoints into main.py:
# =============================================================

import io
import pandas as pd
from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel


class DefectRecord(BaseModel):
    defect_type: str
    defect_size_mm: float
    machine_id: str
    machine_age_years: float
    days_since_maintenance: int
    shift: str
    operator_experience_years: float
    production_speed_mpm: float
    thread_tension_n: float
    humidity_pct: float
    temperature_c: float
    fabric_batch_quality_score: float


@app.post("/api/predict-cause")
async def predict_cause(record: DefectRecord):
    """Predict the root cause of a single defect record."""
    try:
        result = root_cause_model.predict(record.dict())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/predict-cause-batch")
async def predict_cause_batch(file: UploadFile = File(...)):
    """
    Predict root causes for every row in an uploaded CSV.
    Required columns: defect_type, defect_size_mm, machine_id,
    machine_age_years, days_since_maintenance, shift,
    operator_experience_years, production_speed_mpm, thread_tension_n,
    humidity_pct, temperature_c, fabric_batch_quality_score
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file")

    try:
        results = root_cause_model.predict_batch(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    counts = pd.Series([r["predicted_cause"] for r in results]).value_counts().to_dict()
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "cause_counts": counts,
        },
    }

"""
report_enrichment.py
====================
Enriches a basic defect-detection report DataFrame
(Filename / Top Class / Confidence / Status / Timestamp)
into the tabular schema required by the XGBoost root-cause model.

⚠️  WARNING — SIMULATED CONTEXT DATA
    Every production-line context column generated here (machine_id, shift,
    humidity_pct, etc.) is RANDOMLY GENERATED using realistic statistical
    distributions.  These values are NOT real sensor or MES readings.
    The is_simulated_context column in the output is set to True to make
    this explicit.  Replace _random_context_row() with a real MES/SCADA
    data join before using predictions for operational decisions.
"""

import random
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Defect-type vocabulary mapping
# YOLO output class  →  root-cause model's defect_type vocab
# ---------------------------------------------------------------------------
# All classes currently emitted by the YOLOv8 model are mapped below.
# Three categories that exist in the model vocab but are NOT produced by the
# current YOLO model are listed as TODOs — add mappings if/when the YOLO
# model is retrained to detect them.
_DEFECT_TYPE_MAP: dict[str, str] = {
    "hole":         "hole",
    "thread error": "thread_error",
    "objects":      "objects",
    "oil spot":     "oil_spot",
    # TODO: "stain"        not in current YOLO output → add mapping if retrained
    # TODO: "color defect" not in current YOLO output → add mapping if retrained
    # TODO: "cut"          not in current YOLO output → add mapping if retrained
}

_MACHINES  = [f"M{i:02d}" for i in range(1, 13)]   # M01 – M12
_SHIFTS    = ["morning", "afternoon", "night"]
_SHIFT_W   = [0.4, 0.35, 0.25]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _random_context_row() -> dict:
    """
    ⚠️  SIMULATED CONTEXT — NOT real sensor/MES data.
    Distributions are calibrated to produce realistic-looking values for a
    mid-scale textile factory but are entirely synthetic.
    Replace with a real data join (e.g. from SCADA / MES) in production.
    """
    return {
        "defect_size_mm":             round(random.expovariate(1 / 8) + 1, 2),
        "machine_id":                 random.choice(_MACHINES),
        "machine_age_years":          round(random.uniform(0.5, 15), 1),
        "days_since_maintenance":     min(int(random.expovariate(1 / 20)), 180),
        "shift":                      random.choices(_SHIFTS, weights=_SHIFT_W, k=1)[0],
        "operator_experience_years":  round(min(random.expovariate(1 / 4), 25), 1),
        "production_speed_mpm":       round(max(10.0, random.gauss(35, 8)), 1),
        "thread_tension_n":           round(random.gauss(12, 2.5), 2),
        "humidity_pct":               round(random.gauss(55, 12), 1),
        "temperature_c":              round(random.gauss(24, 3), 1),
        "fabric_batch_quality_score": round(_clamp(random.gauss(75, 15), 0, 100), 1),
    }


def enrich_report_for_root_cause(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Transform a batch-predict report DataFrame into the schema expected by
    RootCauseClassifier.predict_batch().

    Parameters
    ----------
    df : DataFrame with columns [Filename, Top Class, Confidence, Status, Timestamp]

    Returns
    -------
    enriched : DataFrame with columns:
        defect_type, defect_size_mm, machine_id, machine_age_years,
        days_since_maintenance, shift, operator_experience_years,
        production_speed_mpm, thread_tension_n, humidity_pct,
        temperature_c, fabric_batch_quality_score,
        is_simulated_context,   ← always True; marks context as synthetic
        Filename, Confidence, Timestamp   ← traceability, not used by model
    warnings : list of unmapped class names encountered (empty if all mapped)

    Notes
    -----
    - Only rows where Status == "DEFECT" are included.
    - Returns (empty DataFrame, []) if no DEFECT rows exist.
    - ⚠️  All production-line context columns are RANDOMLY GENERATED — see
      module docstring.
    """
    defects = df[df["Status"] == "DEFECT"].copy().reset_index(drop=True)

    if defects.empty:
        return pd.DataFrame(), []

    warnings: list[str] = []

    def _map_class(top_class: str) -> str:
        key = top_class.lower().strip()
        if key in _DEFECT_TYPE_MAP:
            return _DEFECT_TYPE_MAP[key]
        # Unmapped class: normalize spaces → underscores and warn
        normalized = key.replace(" ", "_")
        warnings.append(
            f"Unmapped YOLO class '{top_class}' → using '{normalized}' as-is. "
            f"Add an explicit entry to _DEFECT_TYPE_MAP in report_enrichment.py."
        )
        return normalized

    defects["defect_type"] = defects["Top Class"].map(_map_class)

    # ⚠️  SIMULATED CONTEXT — see module-level warning above
    context_df = pd.DataFrame(
        [_random_context_row() for _ in range(len(defects))]
    )

    enriched = pd.concat(
        [defects["defect_type"].reset_index(drop=True),
         context_df.reset_index(drop=True)],
        axis=1,
    )

    # is_simulated_context is always True here — explicit flag so the CSV
    # consumer can never mistake these for real sensor readings
    enriched["is_simulated_context"] = True

    # Traceability columns — not passed to the model, kept for audit trail
    enriched["Filename"]   = defects["Filename"].values
    enriched["Confidence"] = defects["Confidence"].values
    enriched["Timestamp"]  = defects["Timestamp"].values

    return enriched, warnings

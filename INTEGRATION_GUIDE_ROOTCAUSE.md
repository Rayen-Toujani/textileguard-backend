# Model 3 — Root Cause Predictor: Integration Guide

## What this is
A tabular classifier (XGBoost) that predicts **why** a defect happened —
`machine_fault`, `material_issue`, or `human_error` — from production-line
context (machine age, maintenance history, operator experience, shift,
thread tension, humidity, fabric batch quality, etc.).

**This model is trained on synthetic data** — say so explicitly in your PFE
report, exactly as you did for the stain-detection smoke-test images. The
causal relationships (e.g. night shift + low operator experience → higher
human_error probability) were designed from domain logic, not copied from a
real factory. Performance: **72.75% accuracy, Cohen's Kappa 0.57, macro F1
0.72** on a held-out 20% test split — realistic numbers for a genuinely
ambiguous task, not inflated ones.

## Files
| File | Purpose |
|---|---|
| `textile_defect_root_cause.csv` | The synthetic training dataset (4,000 rows) |
| `generate_dataset.py` | Regenerate/tweak the synthetic dataset |
| `train_root_cause_model.py` | Trains RandomForest + XGBoost, saves the best one |
| `root_cause_model.joblib` | The trained model bundle (pipeline + label encoder) |
| `root_cause_classifier.py` | Backend inference module (drop into `backend/`) |
| `main_additions_rootcause.py` | FastAPI endpoints to paste into `main.py` |
| `RootCauseClassifier.tsx` | React frontend component |

## Step 1 — Backend
1. Copy `root_cause_model.joblib` and `root_cause_classifier.py` into `backend/`.
2. In `main.py`, add the import:
   ```python
   from root_cause_classifier import RootCauseClassifier
   ```
3. In your startup event, alongside the YOLOv8 and stain model loading:
   ```python
   root_cause_model = RootCauseClassifier("root_cause_model.joblib")
   ```
4. Paste the two endpoints from `main_additions_rootcause.py`:
   - `POST /api/predict-cause` — single defect record (JSON body)
   - `POST /api/predict-cause-batch` — CSV upload, returns per-row predictions
     plus a cause-count summary

## Step 2 — Frontend
1. Drop `RootCauseClassifier.tsx` into `frontend/src/`.
2. In `App.tsx`: import it, add a third tab button ("Root Cause"), and render
   it — same pattern as the Stain Classifier tab.

## Step 3 — Required CSV columns for batch upload
```
defect_type, defect_size_mm, machine_id, machine_age_years,
days_since_maintenance, shift, operator_experience_years,
production_speed_mpm, thread_tension_n, humidity_pct,
temperature_c, fabric_batch_quality_score
```
`textile_defect_root_cause.csv` is itself a valid input file for testing the
batch endpoint end-to-end.

## Note on model independence
Like the stain classifier, this model doesn't touch `preprocessing.py` — it's
pure tabular data, no images at all. All three models can run side by side
without interfering:
- **Model 1 (YOLOv8):** detects *where* a defect is on the fabric image
- **Model 2 (MobileNetV2):** classifies *whether* a region is stained
- **Model 3 (XGBoost):** predicts *why* the defect likely happened

## Suggested PFE framing
Presenting this as a three-stage pipeline (detect → classify → diagnose) is a
strong narrative for your defense: it moves the project from "can we spot a
defect" to "can we help the factory actually reduce defects," which is the
more compelling quality-control story.

## Honest limitations to state in your report
- Root cause is synthetic-data trained; a real deployment would need actual
  factory logs (MES/SCADA data) paired with defects to validate against.
- Kappa of 0.57 reflects genuine, realistic ambiguity in root-cause
  attribution — even human quality inspectors often disagree on root cause.
  This is a feature of the honest framing, not a bug to hide.

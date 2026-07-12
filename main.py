from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw
import asyncio
import io
import os
import base64
from typing import List, Optional
import pandas as pd
from datetime import datetime

from model import predict_image
from preprocessing import preprocess_single_patch
from stain_classifier import classify_stain, classify_stain_batch, get_stain_model
from root_cause_classifier import RootCauseClassifier
from report_enrichment import enrich_report_for_root_cause
from auth import get_current_user_optional, CurrentUser
from training_data import save_upload_and_prediction

_root_cause_model: "RootCauseClassifier | None" = None


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

app = FastAPI(title="TextileGuard AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_models():
    """Load all ML models in background threads so startup doesn't block port binding."""
    def _load_stain():
        try:
            get_stain_model()
            print("✓ Stain classifier loaded on startup")
        except FileNotFoundError as e:
            print(f"⚠ Stain model not found (Model 2 disabled): {e}")

    def _load_root_cause():
        global _root_cause_model
        try:
            path = os.environ.get("ROOT_CAUSE_MODEL_PATH", "root_cause_model.joblib")
            _root_cause_model = RootCauseClassifier(path)
            print("✓ Root cause classifier loaded on startup")
        except FileNotFoundError as e:
            print(f"⚠ Root cause model not found (Model 3 disabled): {e}")
        except Exception as e:
            print(f"⚠ Root cause model load failed: {e}")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _load_stain)
    loop.run_in_executor(None, _load_root_cause)

def draw_defects_on_image(image: Image.Image, predictions: list) -> Image.Image:
    """Draw circles around detected defects"""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    width, height = image.size
    
    for pred in predictions:
        if pred['class'].lower() == 'good':
            continue
        
        if pred['confidence'] > 0.5:
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 3
            
            color_map = {
                'hole': '#ef4444',
                'oil spot': '#f59e0b',
                'thread error': '#3b82f6',
                'objects': '#8b5cf6'
            }
            color = color_map.get(pred['class'], '#ef4444')
            
            draw.ellipse(
                [center_x - radius, center_y - radius, 
                 center_x + radius, center_y + radius],
                outline=color,
                width=3
            )
            
            label = f"{pred['class']}: {pred['confidence']*100:.1f}%"
            draw.text((10, 10), label, fill=color)
    
    return annotated

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "TextileGuard AI is running", "version": "1.0.0"}

@app.get("/api/health")
def health():
    return {"status": "healthy"}


# TODO(debug): TEMPORARY — remove before final submission. Surfaces the real
# exception from save_upload_and_prediction() instead of only logging it,
# since Render's log dashboard isn't showing anything right now.
@app.post("/api/debug-save-test")
async def debug_save_test(current_user: Optional[CurrentUser] = Depends(get_current_user_optional)):
    if current_user is None:
        return {"error": "No authenticated user — Authorization header missing or invalid"}

    try:
        result = save_upload_and_prediction(
            user_id=current_user.id,
            file_bytes=b"test-image-bytes-not-real",
            original_filename="debug_test.jpg",
            file_type="image",
            model_context="defect_detection",
            model_used="yolo",
            result_json={"debug": True, "test": "manual trigger"},
            confidence=0.99,
            raise_on_error=True,
        )
        return {"success": True, "result": result}
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
):
    """Single image prediction"""
    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents))

        processed_image = preprocess_single_patch(original_image)
        predictions = await asyncio.to_thread(predict_image, processed_image)
        annotated_image = draw_defects_on_image(processed_image, predictions)

        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        top_pred = predictions[0] if predictions else None
        save_upload_and_prediction(
            user_id=current_user.id if current_user else None,
            file_bytes=contents,
            original_filename=file.filename,
            file_type="image",
            model_context="defect_detection",
            model_used="yolo",
            result_json={"predictions": predictions},
            confidence=top_pred["confidence"] if top_pred else None,
        )

        return {
            "predictions": predictions,
            "annotated_image": f"data:image/png;base64,{img_base64}"
        }

    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/batch-predict")
async def batch_predict(
    files: List[UploadFile] = File(...),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
):
    """Batch image prediction"""
    try:
        results = []

        for idx, file in enumerate(files):
            try:
                # Read and process image
                contents = await file.read()
                original_image = Image.open(io.BytesIO(contents))

                processed_image = preprocess_single_patch(original_image)
                predictions = await asyncio.to_thread(predict_image, processed_image)
                annotated_image = draw_defects_on_image(processed_image, predictions)

                # Convert to base64
                buffered = io.BytesIO()
                annotated_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                # Get top prediction
                top_pred = predictions[0] if predictions else {"class": "unknown", "confidence": 0}

                save_upload_and_prediction(
                    user_id=current_user.id if current_user else None,
                    file_bytes=contents,
                    original_filename=file.filename,
                    file_type="image",
                    model_context="defect_detection",
                    model_used="yolo",
                    result_json={"predictions": predictions},
                    confidence=top_pred["confidence"],
                )

                results.append({
                    "filename": file.filename,
                    "index": idx,
                    "predictions": predictions,
                    "annotated_image": f"data:image/png;base64,{img_base64}",
                    "top_class": top_pred['class'],
                    "top_confidence": top_pred['confidence']
                })
                
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "index": idx,
                    "error": str(e)
                })
        
        return {"results": results, "total": len(files)}
        
    except Exception as e:
        import traceback
        print(f"Batch error: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/export-csv")
async def export_csv(results: dict):
    """
    Export batch results as an enriched root-cause CSV for DEFECT rows.
    GOOD rows are excluded — the root-cause model only applies to defects.
    Falls back to the basic 5-column format if all images passed (no defects).
    """
    try:
        data = []
        for result in results.get('results', []):
            if 'error' not in result:
                data.append({
                    'Filename':  result['filename'],
                    'Top Class': result['top_class'],
                    'Confidence': f"{result['top_confidence']*100:.2f}%",
                    'Status':    'DEFECT' if result['top_class'] != 'good' else 'GOOD',
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                })

        df = pd.DataFrame(data)

        enriched, warnings = enrich_report_for_root_cause(df)

        if warnings:
            print(f"[export-csv] Unmapped classes: {warnings}")

        if not enriched.empty:
            # Add root-cause predictions if model is loaded (soft dependency)
            if _root_cause_model is not None:
                try:
                    preds = await asyncio.to_thread(
                        _root_cause_model.predict_batch,
                        enriched[_root_cause_model.required_columns],
                    )
                    enriched["predicted_cause"]  = [p["predicted_cause"] for p in preds]
                    enriched["cause_confidence"] = [round(p["confidence"], 4) for p in preds]
                except Exception as e:
                    print(f"[export-csv] Root cause prediction skipped: {e}")
            out_df   = enriched
            filename = "root_cause_report.csv"
        else:
            # No DEFECT rows — return the plain report so the download still works
            out_df   = df
            filename = "textile_analysis_report.csv"

        csv_buffer = io.StringIO()
        out_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/classify-stain")
async def classify_stain_endpoint(
    file: UploadFile = File(...),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
):
    """
    Binary fabric stain classification.
    Returns whether the fabric is defect-free or has a stain/defect.

    Response:
        {
            "filename":         str,
            "label":            "defect_free" | "defect",
            "probability":      float,
            "confidence":       float,
            "confidence_label": str,
            "threshold":        float,
            "passed":           bool
        }
    """
    try:
        image_bytes = await file.read()
        result = await asyncio.to_thread(classify_stain, image_bytes)
        result["filename"] = file.filename

        save_upload_and_prediction(
            user_id=current_user.id if current_user else None,
            file_bytes=image_bytes,
            original_filename=file.filename,
            file_type="image",
            model_context="stain_classifier",
            model_used="stain",
            result_json=result,
            confidence=result["confidence"],
        )

        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stain classification failed: {str(e)}")

@app.post("/api/classify-stain-batch")
async def classify_stain_batch_endpoint(
    files: List[UploadFile] = File(...),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
):
    """
    Batch stain classification — processes all images in one model call.

    Response:
        {
            "results": [...],      # one result per image
            "summary": {
                "total":        int,
                "passed":       int,
                "failed":       int,
                "pass_rate":    float
            }
        }
    """
    try:
        images = [(f.filename, await f.read()) for f in files]
        results = await asyncio.to_thread(classify_stain_batch, images)

        for (filename, image_bytes), result in zip(images, results):
            save_upload_and_prediction(
                user_id=current_user.id if current_user else None,
                file_bytes=image_bytes,
                original_filename=filename,
                file_type="image",
                model_context="stain_classifier",
                model_used="stain",
                result_json=result,
                confidence=result["confidence"],
            )

        passed   = sum(1 for r in results if r["passed"])
        failed   = len(results) - passed
        pass_rate = round(passed / len(results) * 100, 1) if results else 0

        return {
            "results": results,
            "summary": {
                "total":     len(results),
                "passed":    passed,
                "failed":    failed,
                "pass_rate": pass_rate,
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")



@app.post("/api/enrich-and-predict-cause")
async def enrich_and_predict_cause(results: dict):
    """
    One-shot pipeline: batch-predict report → enrich → root-cause prediction.

    Accepts the same `results` payload as /api/export-csv.
    Returns an enriched CSV (download) with predicted_cause and cause_confidence
    appended to each DEFECT row.

    ⚠️  Production-line context columns (machine_id, shift, humidity_pct, etc.)
    are RANDOMLY GENERATED placeholders — is_simulated_context=True marks them
    in every output row.  Wire up real MES/SCADA data before using predictions
    for operational decisions.
    """
    if _root_cause_model is None:
        raise HTTPException(status_code=503, detail="Root cause model not loaded")

    data = []
    for result in results.get("results", []):
        if "error" not in result:
            data.append({
                "Filename":  result["filename"],
                "Top Class": result["top_class"],
                "Confidence": f"{result['top_confidence'] * 100:.2f}%",
                "Status":    "DEFECT" if result["top_class"] != "good" else "GOOD",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    if not data:
        raise HTTPException(status_code=400, detail="No results to process")

    report_df = pd.DataFrame(data)
    enriched, warnings = enrich_report_for_root_cause(report_df)

    if enriched.empty:
        raise HTTPException(
            status_code=400,
            detail="No DEFECT rows found — all images passed quality check",
        )

    if warnings:
        print(f"[enrich-and-predict-cause] Unmapped classes: {warnings}")

    try:
        predictions = await asyncio.to_thread(
            _root_cause_model.predict_batch,
            enriched[_root_cause_model.required_columns],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Root cause prediction failed: {str(e)}")

    enriched["predicted_cause"]   = [p["predicted_cause"] for p in predictions]
    enriched["cause_confidence"]  = [round(p["confidence"], 4) for p in predictions]

    csv_buffer = io.StringIO()
    enriched.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return StreamingResponse(
        iter([csv_buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=root_cause_report.csv"},
    )


@app.post("/api/predict-cause")
async def predict_cause(
    record: DefectRecord,
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
):
    """Predict the root cause of a single defect record (JSON body)."""
    if _root_cause_model is None:
        raise HTTPException(status_code=503, detail="Root cause model not loaded")
    try:
        result = await asyncio.to_thread(_root_cause_model.predict, record.model_dump())

        save_upload_and_prediction(
            user_id=current_user.id if current_user else None,
            file_bytes=None,
            original_filename=None,
            file_type=None,
            model_context=None,
            model_used="root_cause",
            result_json=result,
            confidence=result.get("confidence"),
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Root cause prediction failed: {str(e)}")


@app.post("/api/predict-cause-batch")
async def predict_cause_batch(
    file: UploadFile = File(...),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
):
    """
    Predict root causes for every row in an uploaded CSV.
    Required columns: defect_type, defect_size_mm, machine_id, machine_age_years,
    days_since_maintenance, shift, operator_experience_years, production_speed_mpm,
    thread_tension_n, humidity_pct, temperature_c, fabric_batch_quality_score
    """
    if _root_cause_model is None:
        raise HTTPException(status_code=503, detail="Root cause model not loaded")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file")

    try:
        results = await asyncio.to_thread(_root_cause_model.predict_batch, df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch root cause prediction failed: {str(e)}")

    for result in results:
        save_upload_and_prediction(
            user_id=current_user.id if current_user else None,
            file_bytes=None,
            original_filename=None,
            file_type=None,
            model_context=None,
            model_used="root_cause",
            result_json=result,
            confidence=result.get("confidence"),
        )

    counts = pd.Series([r["predicted_cause"] for r in results]).value_counts().to_dict()
    return {
        "results": results,
        "summary": {"total": len(results), "cause_counts": counts},
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw
import io
import os
import base64
from typing import List
import pandas as pd
from datetime import datetime

from model import predict_image
from preprocessing import preprocess_single_patch
from stain_classifier import classify_stain, classify_stain_batch, get_stain_model

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
async def load_stain_model():
    """Pre-load stain model on startup so first request isn't slow."""
    try:
        get_stain_model()
        print("✓ Stain classifier loaded on startup")
    except FileNotFoundError as e:
        print(f"⚠ Stain model not found (Part 2 disabled): {e}")

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

@app.get("/")
def root():
    return {"status": "TextileGuard AI is running", "version": "1.0.0"}

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Single image prediction"""
    try:
        contents = await file.read()
        original_image = Image.open(io.BytesIO(contents))
        
        processed_image = preprocess_single_patch(original_image)
        predictions = predict_image(processed_image)
        annotated_image = draw_defects_on_image(processed_image, predictions)
        
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
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
async def batch_predict(files: List[UploadFile] = File(...)):
    """Batch image prediction"""
    try:
        results = []
        
        for idx, file in enumerate(files):
            try:
                # Read and process image
                contents = await file.read()
                original_image = Image.open(io.BytesIO(contents))
                
                processed_image = preprocess_single_patch(original_image)
                predictions = predict_image(processed_image)
                annotated_image = draw_defects_on_image(processed_image, predictions)
                
                # Convert to base64
                buffered = io.BytesIO()
                annotated_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Get top prediction
                top_pred = predictions[0] if predictions else {"class": "unknown", "confidence": 0}
                
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
    """Export batch results as CSV"""
    try:
        data = []
        for result in results.get('results', []):
            if 'error' not in result:
                data.append({
                    'Filename': result['filename'],
                    'Top Class': result['top_class'],
                    'Confidence': f"{result['top_confidence']*100:.2f}%",
                    'Status': 'DEFECT' if result['top_class'] != 'good' else 'GOOD',
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        df = pd.DataFrame(data)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=textile_analysis_report.csv"}
        )
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/classify-stain")
async def classify_stain_endpoint(file: UploadFile = File(...)):
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
        result = classify_stain(image_bytes)
        result["filename"] = file.filename
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stain classification failed: {str(e)}")

@app.post("/api/classify-stain-batch")
async def classify_stain_batch_endpoint(files: List[UploadFile] = File(...)):
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
        results = classify_stain_batch(images)

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



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
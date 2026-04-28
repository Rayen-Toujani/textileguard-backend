from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

from model import predict_image
from preprocessing import extract_patches

app = FastAPI(title="TextileGuard AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "TextileGuard AI is running", "version": "1.0.0"}

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract 64x64 patches
        patches = extract_patches(image, patch_size=64)
        
        # Predict on each patch
        all_predictions = []
        for i, patch in enumerate(patches):
            patch_preds = predict_image(patch)
            if patch_preds:
                # Add patch index to results
                for pred in patch_preds:
                    pred['patch_index'] = i
                all_predictions.extend(patch_preds)
        
        # Aggregate results (take highest confidence prediction)
        if all_predictions:
            best_pred = max(all_predictions, key=lambda x: x['confidence'])
            return {"predictions": [best_pred], "total_patches": len(patches)}
        
        return {"predictions": [], "total_patches": len(patches)}
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
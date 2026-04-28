from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

from model import predict_image
from preprocessing import preprocess_single_patch

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
        # Read file contents
        contents = await file.read()
        
        # Open as PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Apply TILDA-style preprocessing:
        # - Convert to grayscale
        # - Resize to 64x64
        # - Auto-contrast normalization
        # - Convert back to RGB for YOLO
        processed_image = preprocess_single_patch(image)
        
        # Pass preprocessed image to model
        predictions = predict_image(processed_image)
        
        return {"predictions": predictions}
        
    except Exception as e:
        import traceback
        print(f"Error in predict endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
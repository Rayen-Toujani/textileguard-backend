from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw
import io
import os
import base64

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

def draw_defects_on_image(image: Image.Image, predictions: list) -> Image.Image:
    """
    Draw circles around detected defects on the image
    """
    # Create a copy to draw on
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Get image dimensions
    width, height = image.size
    
    for pred in predictions:
        # Skip "good" class - only highlight defects
        if pred['class'].lower() == 'good':
            continue
        
        # For 64x64 patch, draw circle in center
        # If confidence is high enough to be considered a defect
        if pred['confidence'] > 0.5:
            # Draw circle in center of image
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 3
            
            # Color based on defect type
            color_map = {
                'hole': '#ef4444',          # Red
                'oil spot': '#f59e0b',      # Orange
                'thread error': '#3b82f6',  # Blue
                'objects': '#8b5cf6'        # Purple
            }
            color = color_map.get(pred['class'], '#ef4444')
            
            # Draw circle
            draw.ellipse(
                [center_x - radius, center_y - radius, 
                 center_x + radius, center_y + radius],
                outline=color,
                width=3
            )
            
            # Draw label
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
    try:
        # Read file contents
        contents = await file.read()
        
        # Open as PIL Image
        original_image = Image.open(io.BytesIO(contents))
        
        # Apply TILDA-style preprocessing for prediction
        processed_image = preprocess_single_patch(original_image)
        
        # Get predictions
        predictions = predict_image(processed_image)
        
        # Draw defects on the processed image (64x64)
        annotated_image = draw_defects_on_image(processed_image, predictions)
        
        # Convert annotated image to base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "predictions": predictions,
            "annotated_image": f"data:image/png;base64,{img_base64}"
        }
        
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
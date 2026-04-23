from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")

def predict_image(image):
    results = model(image)
    
    # For classification models
    probs = results[0].probs
    
    if probs is None:
        # This might be a detection model, not classification
        # Try detection format instead
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        
        predictions = []
        for box in boxes:
            predictions.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xywh[0].cpu().numpy().tolist()  # Convert tensor to list
            })
        return predictions
    
    # Classification model predictions
    predictions = []
    for i in probs.top5:
        predictions.append({
            "class": model.names[int(i)],
            "confidence": float(probs.data[int(i)].cpu().numpy())  # Convert to Python float
        })
    
    return predictions
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np

model = YOLO("best.pt")

def predict_image(image):
    try:
        # Ensure image is PIL Image (not numpy array)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            # If it's something else, try to convert
            image = Image.open(image)
        
        # Run inference
        results = model(image, verbose=False)
        result = results[0]
        
        predictions = []
        
        # Classification model
        if result.probs is not None:
            probs = result.probs
            top5_indices = probs.top5
            
            for idx in top5_indices:
                idx = int(idx)
                conf_value = probs.data[idx]
                
                if isinstance(conf_value, torch.Tensor):
                    conf = float(conf_value.item())
                else:
                    conf = float(conf_value)
                
                predictions.append({
                    "class": str(model.names[idx]),
                    "confidence": conf
                })
        
        # Detection model
        elif result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_tensor = box.cls[0]
                cls_id = int(cls_tensor.item() if isinstance(cls_tensor, torch.Tensor) else cls_tensor)
                
                conf_tensor = box.conf[0]
                conf = float(conf_tensor.item() if isinstance(conf_tensor, torch.Tensor) else conf_tensor)
                
                bbox_tensor = box.xywh[0]
                if isinstance(bbox_tensor, torch.Tensor):
                    bbox = bbox_tensor.cpu().tolist()
                else:
                    bbox = bbox_tensor.tolist()
                
                predictions.append({
                    "class": str(model.names[cls_id]),
                    "confidence": conf,
                    "bbox": bbox
                })
        
        return predictions
        
    except Exception as e:
        import traceback
        print(f"Error in predict_image: {str(e)}")
        print(traceback.format_exc())
        return []
from ultralytics import YOLO
import torch

model = YOLO("best.pt")

def predict_image(image):
    try:
        # Run inference
        results = model(image, verbose=False)
        result = results[0]
        
        predictions = []
        
        # Classification model
        if result.probs is not None:
            probs = result.probs
            top5_indices = probs.top5  # Already a list or tensor
            
            for idx in top5_indices:
                idx = int(idx)
                conf_value = probs.data[idx]
                
                # Handle both tensor and numpy
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
                # Get class
                cls_tensor = box.cls[0]
                cls_id = int(cls_tensor.item() if isinstance(cls_tensor, torch.Tensor) else cls_tensor)
                
                # Get confidence
                conf_tensor = box.conf[0]
                conf = float(conf_tensor.item() if isinstance(conf_tensor, torch.Tensor) else conf_tensor)
                
                # Get bbox (center x, y, width, height)
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
        # Log error for debugging
        import traceback
        print(f"Error in predict_image: {str(e)}")
        print(traceback.format_exc())
        return []
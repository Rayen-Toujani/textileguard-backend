from ultralytics import YOLO
import torch
import os
import tempfile

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = YOLO("best.pt")
    return _model

def predict_image(pil_image):
    """
    Expects a PIL Image object
    """
    try:
        # Save PIL image to temporary file
        # Ultralytics has issues with PIL Image objects in classification mode
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            pil_image.save(tmp.name, format='JPEG')
            tmp_path = tmp.name
        
        try:
            # Run inference with file path instead of PIL Image
            results = _get_model()(tmp_path, verbose=False)
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
                        "class": str(_get_model().names[idx]),
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
                        "class": str(_get_model().names[cls_id]),
                        "confidence": conf,
                        "bbox": bbox
                    })
            
            print(f"Predictions: {predictions}")
            return predictions
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
    except Exception as e:
        import traceback
        print(f"Error in predict_image: {str(e)}")
        print(traceback.format_exc())
        return []
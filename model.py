from ultralytics import YOLO

model = YOLO("best.pt")

def predict_image(image):
    results = model(image)
    probs = results[0].probs

    if probs is None:
        return []

    predictions = []
    for i in probs.top5:
        predictions.append({
            "class": model.names[int(i)],
            "confidence": float(probs.data[int(i)])
        })

    return predictions

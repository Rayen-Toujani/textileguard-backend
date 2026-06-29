"""
TextileGuard AI — Part 2
Fabric stain binary classifier using MobileNetV2 (TFLite runtime)
Integrates with existing FastAPI backend (main.py)
"""

import io
import threading
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import os
import logging

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("STAIN_MODEL_PATH", "stain_model.tflite")
IMG_SIZE        = (224, 224)
THRESHOLD       = 0.075   # optimal threshold from training (Kappa 0.9467)
CLASS_NAMES     = {0: "defect_free", 1: "defect"}
CONFIDENCE_LABELS = {
    (0.00, 0.30): "High confidence — clean fabric",
    (0.30, 0.50): "Low confidence — likely clean",
    (0.50, 0.70): "Low confidence — likely defect",
    (0.70, 1.00): "High confidence — defect detected",
}

# ── Model loader (singleton) ──────────────────────────────────────────────────
_interpreter = None
_model_lock = threading.Lock()
# TFLite Interpreter instances aren't safe for concurrent invoke() calls, so
# every set_tensor/invoke/get_tensor sequence below is serialized through this.
_inference_lock = threading.Lock()

def get_stain_model():
    """Load the TFLite interpreter once and reuse across requests. Thread-safe against concurrent first calls."""
    global _interpreter
    if _interpreter is None:
        with _model_lock:
            if _interpreter is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Stain model not found at '{MODEL_PATH}'. "
                        f"Set STAIN_MODEL_PATH env var or place stain_model.tflite in backend/."
                    )
                logger.info(f"Loading stain classifier from {MODEL_PATH} ...")
                interpreter = Interpreter(model_path=MODEL_PATH)
                interpreter.allocate_tensors()
                _interpreter = interpreter
                logger.info("Stain classifier loaded ✓")
    return _interpreter


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes → model-ready (1, 224, 224, 3) float32 array.
    Keeps colour — stain detection depends on colour information.
    Does NOT apply the grayscale/patch pipeline from preprocessing.py
    because MobileNetV2 was trained on full RGB images.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


def _run_inference(batch: np.ndarray) -> np.ndarray:
    """Run the TFLite interpreter on a (N, 224, 224, 3) batch, returning N probabilities."""
    interpreter = get_stain_model()
    with _inference_lock:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if tuple(input_details[0]["shape"]) != batch.shape:
            interpreter.resize_tensor_input(input_details[0]["index"], list(batch.shape))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], batch)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]["index"]).flatten()


def _build_result(prob: float) -> dict:
    label  = CLASS_NAMES[int(prob >= THRESHOLD)]
    passed = prob < THRESHOLD
    confidence = abs(prob - 0.5) * 2   # 0.0 = uncertain, 1.0 = certain

    conf_label = "Uncertain"
    for (lo, hi), text in CONFIDENCE_LABELS.items():
        if lo <= prob < hi:
            conf_label = text
            break

    return {
        "label":           label,
        "probability":     round(prob, 4),
        "confidence":      round(confidence, 4),
        "confidence_label": conf_label,
        "threshold":       THRESHOLD,
        "passed":          passed,
    }


# ── Inference ─────────────────────────────────────────────────────────────────
def classify_stain(image_bytes: bytes) -> dict:
    """
    Run stain classification on raw image bytes.

    Returns:
        {
            "label":       "defect_free" | "defect",
            "probability": float,          # raw model output 0.0–1.0
            "confidence":  float,          # distance from decision boundary
            "confidence_label": str,       # human-readable confidence
            "threshold":   float,          # threshold used
            "passed":      bool            # True = defect_free
        }
    """
    arr  = preprocess_image(image_bytes)
    prob = float(_run_inference(arr)[0])
    return _build_result(prob)


# ── Batch inference ───────────────────────────────────────────────────────────
def classify_stain_batch(images: list[tuple[str, bytes]]) -> list[dict]:
    """
    Classify multiple images efficiently in one model call.

    Args:
        images: list of (filename, image_bytes) tuples

    Returns:
        list of result dicts (same shape as classify_stain), each with
        an added "filename" key
    """
    if not images:
        return []

    batch = np.concatenate([preprocess_image(b) for _, b in images], axis=0)
    probs = _run_inference(batch)

    results = []
    for (filename, _), prob in zip(images, probs):
        result = _build_result(float(prob))
        result["filename"] = filename
        results.append(result)

    return results

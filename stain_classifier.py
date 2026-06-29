"""
TextileGuard AI — Part 2
Fabric stain binary classifier using MobileNetV2
Integrates with existing FastAPI backend (main.py)
"""

import io
import threading
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import logging

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("STAIN_MODEL_PATH", "stain_model.h5")
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
_model = None
_model_lock = threading.Lock()

def get_stain_model():
    """Load model once and reuse across requests. Thread-safe against concurrent first calls."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Stain model not found at '{MODEL_PATH}'. "
                        f"Set STAIN_MODEL_PATH env var or place stain_model.h5 in backend/."
                    )
                logger.info(f"Loading stain classifier from {MODEL_PATH} ...")
                _model = load_model(MODEL_PATH)
                logger.info("Stain classifier loaded ✓")
    return _model


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
    model = get_stain_model()
    arr   = preprocess_image(image_bytes)

    prob  = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[int(prob >= THRESHOLD)]
    passed = prob < THRESHOLD

    # Confidence = how far the probability is from the 0.5 midpoint
    confidence = abs(prob - 0.5) * 2   # 0.0 = uncertain, 1.0 = certain

    # Human-readable confidence label
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

    model = get_stain_model()

    batch  = np.concatenate([preprocess_image(b) for _, b in images], axis=0)
    probs  = model.predict(batch, verbose=0).flatten()

    results = []
    for (filename, _), prob in zip(images, probs):
        prob   = float(prob)
        label  = CLASS_NAMES[int(prob >= THRESHOLD)]
        passed = prob < THRESHOLD
        conf   = abs(prob - 0.5) * 2

        conf_label = "Uncertain"
        for (lo, hi), text in CONFIDENCE_LABELS.items():
            if lo <= prob < hi:
                conf_label = text
                break

        results.append({
            "filename":         filename,
            "label":            label,
            "probability":      round(prob, 4),
            "confidence":       round(conf, 4),
            "confidence_label": conf_label,
            "threshold":        THRESHOLD,
            "passed":           passed,
        })

    return results
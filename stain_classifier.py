"""
TextileGuard AI — Part 2
Fabric stain binary classifier using MobileNetV2 (Keras H5)
Integrates with existing FastAPI backend (main.py)
"""

import io
import threading
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.environ.get("STAIN_MODEL_PATH", "stain_model.h5")
IMG_SIZE        = (224, 224)
THRESHOLD       = 0.075   # optimal threshold from training (Kappa 0.9467)
CLASS_NAMES     = {0: "defect_free", 1: "defect"}

# ── Model loader (singleton) ──────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()

def get_stain_model():
    """Load the Keras model once and reuse across requests. Thread-safe against concurrent first calls."""
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
                import tensorflow as tf
                _model = tf.keras.models.load_model(MODEL_PATH)
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


def _run_inference(batch: np.ndarray) -> np.ndarray:
    """Run the Keras model on a (N, 224, 224, 3) batch, returning N probabilities."""
    model = get_stain_model()
    return model.predict(batch, verbose=0).flatten()


def _build_result(prob: float) -> dict:
    is_defect = prob >= THRESHOLD
    label  = CLASS_NAMES[int(is_defect)]
    passed = not is_defect

    # Distance from the actual decision boundary (THRESHOLD), normalized to
    # each side's range, so confidence reflects how close prob is to the
    # boundary that actually decides the label — not a fixed 0.5 midpoint.
    if is_defect:
        confidence = (prob - THRESHOLD) / max(1.0 - THRESHOLD, 1e-6)
    else:
        confidence = (THRESHOLD - prob) / max(THRESHOLD, 1e-6)

    if is_defect:
        conf_label = "High confidence — defect detected" if confidence >= 0.5 else "Low confidence — likely defect"
    else:
        conf_label = "High confidence — clean fabric" if confidence >= 0.5 else "Low confidence — likely clean"

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

"""
Garment detection and cropping using a YOLO11s model fine-tuned on a
DeepFashion-derived dataset.

Part 3 of TextileGuard AI: given a frame, locate clothing regions and
return cropped images for each detected garment so they can be fed into
downstream defect analysis.

Classes (11): dress, long_sleeved_dress, long_sleeved_outwear,
long_sleeved_shirt, short_sleeved_outwear, short_sleeved_shirt, shorts,
skirt, trousers, vest, vest_dress.

Validation performance (full model): mAP50 0.5845, mAP50-95 0.5078,
Precision 0.6559, Recall 0.5623. short_sleeved_outwear performs
noticeably worse than the other classes (low training instance count),
with long_sleeved_dress and vest also weaker than average -- treat
detections in those classes with more skepticism at a given
conf_threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "bestYOLO11model.pt"


class GarmentDetector:
    """
    Wraps the fine-tuned YOLO11s garment detection model.

    The model is loaded once when a GarmentDetector is constructed and
    reused across calls to detect_and_crop -- construct one instance at
    startup and hold onto it rather than creating a new instance per
    request.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        logger.info("Loading garment detector weights from %s", self.model_path)
        self.model = YOLO(str(self.model_path))
        self.class_names: dict[int, str] = self.model.names

    def detect_and_crop(
        self, image: np.ndarray, conf_threshold: float = 0.5, imgsz: int = 640
    ) -> list[dict[str, Any]]:
        """
        Run garment detection on a single image and crop each detected region.

        Args:
            image: image as a numpy array (H, W, 3), e.g. a frame read via
                cv2.VideoCapture/cv2.imread (BGR) -- channel order is
                preserved as-is into the crop.
            conf_threshold: minimum detection confidence to keep a result.
            imgsz: side length (px) ultralytics resizes the image to before
                inference. Was temporarily dropped to 384 to try to fit a
                production OOM -- reverted to ultralytics' default 640 once
                the real root cause turned out to be baseline/idle memory
                (four models resident at startup), not per-inference size;
                384 measurably hurt recall (0.956 -> 0.376 confidence on a
                reference test image, enough to fall below this endpoint's
                default conf_threshold=0.5 and silently miss the detection)
                for no benefit on the actual problem. bbox/crop coordinates
                are always returned in the original image's pixel space
                regardless of imgsz.

        Returns:
            A list of dicts, one per detection above conf_threshold:
                {
                    "class_name": str,
                    "confidence": float,
                    "bbox": (x1, y1, x2, y2),       # ints, pixel coords in `image`
                    "cropped_image": np.ndarray,    # image[y1:y2, x1:x2], same channel order as `image`
                }
        """
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty numpy array")

        results = self.model.predict(image, conf=conf_threshold, imgsz=imgsz, verbose=False)
        result = results[0]

        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections

        img_h, img_w = image.shape[:2]

        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(img_w, int(round(x2)))
            y2 = min(img_h, int(round(y2)))
            if x2 <= x1 or y2 <= y1:
                # Degenerate box after clamping to image bounds; skip it.
                continue

            detections.append(
                {
                    "class_name": self.class_names.get(cls_id, str(cls_id)),
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "cropped_image": image[y1:y2, x1:x2].copy(),
                }
            )

        return detections


def _sanity_check(image_path: str, conf_threshold: float = 0.5) -> None:
    """Load the detector, run it on a single test image, and print results."""
    import cv2

    detector = GarmentDetector()
    print(f"Loaded model from: {detector.model_path}")
    print(f"Classes ({len(detector.class_names)}): {list(detector.class_names.values())}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    print(f"Test image: {image_path} shape={image.shape}")

    detections = detector.detect_and_crop(image, conf_threshold=conf_threshold)

    print(f"\n{len(detections)} detection(s) above conf_threshold={conf_threshold}:")
    for i, det in enumerate(detections):
        print(
            f"  [{i}] {det['class_name']:<24} conf={det['confidence']:.3f} "
            f"bbox={det['bbox']} crop_shape={det['cropped_image'].shape}"
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python garment_detector.py <path_to_test_image> [conf_threshold]")
        sys.exit(1)

    _sanity_check(
        image_path=sys.argv[1],
        conf_threshold=float(sys.argv[2]) if len(sys.argv) > 2 else 0.5,
    )

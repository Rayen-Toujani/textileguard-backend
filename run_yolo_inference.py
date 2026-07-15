"""
Standalone subprocess entry point for running a single YOLO inference call
in an isolated process that exits when done.

Why: PyTorch's CPU tensor allocations are backed by glibc malloc, which
does not reliably return freed heap memory to the OS within a long-lived
process. Verified in production: dropping a model's Python reference and
calling gc.collect() (in-process "unload") left RSS elevated enough that
loading the second model on top of it still OOM'd on Render's 512MB
ceiling. Process exit is the one thing guaranteed to reclaim everything,
so main.py now runs each YOLO call here instead of holding either model
in the main FastAPI process — see main.py's _run_inference_subprocess().

Usage:
    python run_yolo_inference.py --model {defect,garment} --image <path> --output <path> [--conf 0.5] [--imgsz 640]

Writes a JSON result to --output and exits 0 on success. On failure,
prints the error to stderr and exits non-zero — callers should treat any
non-zero exit as failure and not attempt to read --output.

Output shapes:
    --model defect: {"predictions": [{"class": str, "confidence": float, "bbox": [...] (optional)}, ...]}
    --model garment: {"detections": [{"class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2],
                                       "cropped_image_b64": str | None}, ...]}
"""

import argparse
import base64
import json
import sys


def run_defect(image_path: str) -> dict:
    from PIL import Image
    from ultralytics import YOLO

    from model import predict_image

    model = YOLO("best.pt")
    predictions = predict_image(Image.open(image_path), model)
    return {"predictions": predictions}


def run_garment(image_path: str, conf: float, imgsz: int) -> dict:
    import cv2

    from garment_detector import GarmentDetector

    detector = GarmentDetector()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    detections = detector.detect_and_crop(image, conf_threshold=conf, imgsz=imgsz)

    out = []
    for det in detections:
        ok, buf = cv2.imencode(".jpg", det["cropped_image"])
        out.append(
            {
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": list(det["bbox"]),
                "cropped_image_b64": base64.b64encode(buf.tobytes()).decode("ascii") if ok else None,
            }
        )
    return {"detections": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["defect", "garment"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    try:
        if args.model == "defect":
            result = run_defect(args.image)
        else:
            result = run_garment(args.image, args.conf, args.imgsz)

        with open(args.output, "w") as f:
            json.dump(result, f)

    except Exception as e:
        import traceback

        print(f"run_yolo_inference ({args.model}) failed: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

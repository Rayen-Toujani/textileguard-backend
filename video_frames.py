"""
Video frame extraction and de-duplication for TextileGuard AI.

Part 3 pipeline (frame sampling stage): pull frames from a video at a
fixed time interval, then drop frames that are near-duplicates of the
last kept frame, so garment detection isn't run repeatedly against
essentially the same shot (e.g. a static camera or a paused subject).
"""

from __future__ import annotations

import logging
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SIMILARITY_DOWNSCALE_SIZE = (64, 64)


def extract_frames(video_path: str, interval_seconds: float = 2.5) -> Iterator[tuple[np.ndarray, int, float]]:
    """
    Sample frames from a video at a fixed time interval.

    This is a generator: frames are read and yielded one at a time rather
    than collected into a list up front, so memory use stays flat
    regardless of video length. Wrap the call in list(...) if you need
    all frames materialized at once.

    Args:
        video_path: path to a video file readable by OpenCV.
        interval_seconds: seconds between sampled frames.

    Yields:
        (frame, frame_index, timestamp_seconds) tuples, in video order:
          - frame: a BGR frame as a numpy array (H, W, 3)
          - frame_index: the frame's 0-based position in the source video
          - timestamp_seconds: frame_index / fps
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0  # fallback for streams with missing/unreliable metadata
        frame_interval = max(1, round(fps * interval_seconds))

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_interval == 0:
                yield frame, frame_index, frame_index / fps

            frame_index += 1
    finally:
        cap.release()


def frames_are_similar(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.95) -> bool:
    """
    Cheap similarity check between two frames.

    Downscales both frames to a small fixed size, converts to grayscale,
    and compares via normalized cross-correlation template matching
    (cv2.matchTemplate, TM_CCOEFF_NORMED). This is much cheaper than a
    perceptual hash or full SSIM and is good enough to catch near-duplicate
    consecutive frames.

    Args:
        frame1: image as a numpy array (grayscale or BGR).
        frame2: image as a numpy array (grayscale or BGR).
        threshold: similarity score (roughly 0-1) at or above which the
            frames are considered duplicates.

    Returns:
        True if the frames are similar enough to be considered duplicates.
    """
    small1 = _to_small_gray(frame1)
    small2 = _to_small_gray(frame2)

    score = float(cv2.matchTemplate(small1, small2, cv2.TM_CCOEFF_NORMED)[0][0])
    return score >= threshold


def _to_small_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(frame, _SIMILARITY_DOWNSCALE_SIZE, interpolation=cv2.INTER_AREA)


def extract_distinct_frames(
    video_path: str,
    interval_seconds: float = 2.5,
    similarity_threshold: float = 0.95,
) -> tuple[list[tuple[np.ndarray, int, float]], int]:
    """
    Sample frames at a fixed interval and keep only the ones that are
    visually distinct from the last kept frame.

    Args:
        video_path: path to a video file readable by OpenCV.
        interval_seconds: seconds between sampled frames (see extract_frames).
        similarity_threshold: passed to frames_are_similar; a sampled frame
            scoring at or above this against the last kept frame is
            dropped as a near-duplicate.

    Returns:
        (kept_frames, total_sampled):
          - kept_frames: list of (frame, frame_index, timestamp_seconds)
            tuples that survived de-duplication, in video order.
          - total_sampled: count of raw frames sampled at interval_seconds
            before de-duplication.
    """
    kept_frames: list[tuple[np.ndarray, int, float]] = []
    last_kept_frame: np.ndarray | None = None
    total_sampled = 0

    for frame, frame_index, timestamp_seconds in extract_frames(video_path, interval_seconds=interval_seconds):
        total_sampled += 1
        if last_kept_frame is None or not frames_are_similar(frame, last_kept_frame, threshold=similarity_threshold):
            kept_frames.append((frame, frame_index, timestamp_seconds))
            last_kept_frame = frame

    logger.info(
        "extract_distinct_frames: kept %d/%d frame(s) from %s",
        len(kept_frames), total_sampled, video_path,
    )
    return kept_frames, total_sampled


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_frames.py <path_to_video> [interval_seconds] [similarity_threshold]")
        sys.exit(1)

    video_path_arg = sys.argv[1]
    interval_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
    similarity_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95

    distinct_frames, total_sampled = extract_distinct_frames(
        video_path_arg, interval_seconds=interval_arg, similarity_threshold=similarity_arg
    )
    print(f"Sampled {total_sampled} raw frame(s), kept {len(distinct_frames)} distinct frame(s) from {video_path_arg}")
    for i, (frame, frame_index, timestamp_seconds) in enumerate(distinct_frames):
        print(f"  [{i}] frame_index={frame_index} t={timestamp_seconds:.2f}s shape={frame.shape}")

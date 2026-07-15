"""
Part 2 — silently persists predictions (and their source file, if any) for
future model retraining, tied to whichever user happened to be logged in.

Prediction endpoints stay PUBLIC: this module is a best-effort side effect
that must never affect the response the caller is waiting on.
"""

import logging
import mimetypes
import time
from typing import Optional
from uuid import uuid4

from supabase_client import get_supabase

logger = logging.getLogger(__name__)

UPLOADS_BUCKET = "uploads"


def save_upload(
    user_id: Optional[str],
    file_bytes: bytes,
    original_filename: Optional[str],
    file_type: Optional[str],
    model_context: Optional[str],
    raise_on_error: bool = False,
) -> Optional[str]:
    """
    Upload file_bytes to the "uploads" bucket at {user_id}/{uuid}.{ext} and
    insert an `uploads` row. Returns the new upload_id, or None if user_id
    is None (anonymous — no-op) or the write failed.

    By default, never raises — errors are logged and swallowed. Pass
    raise_on_error=True (debug-only) to propagate the exception instead.
    """
    if user_id is None:
        return None

    try:
        client = get_supabase()

        extension = ""
        if original_filename and "." in original_filename:
            extension = original_filename.rsplit(".", 1)[1].lower()
        storage_path = (
            f"{user_id}/{uuid4()}.{extension}" if extension else f"{user_id}/{uuid4()}"
        )
        content_type = mimetypes.guess_type(original_filename or "")[0] or "application/octet-stream"

        # TEMPORARY: split [TIMING] logging to diagnose a production timeout
        # on /api/analyze-video -- remove once the slow stage is identified.
        t0 = time.time()
        client.storage.from_(UPLOADS_BUCKET).upload(
            storage_path,
            file_bytes,
            {"content-type": content_type},
        )
        print(f"[TIMING] save_upload.storage_upload: {time.time() - t0:.2f}s")

        t0 = time.time()
        upload_row = (
            client.table("uploads")
            .insert(
                {
                    "user_id": user_id,
                    "storage_path": storage_path,
                    "file_type": file_type,
                    "original_filename": original_filename,
                    "model_context": model_context,
                }
            )
            .execute()
        )
        print(f"[TIMING] save_upload.table_insert: {time.time() - t0:.2f}s")
        return upload_row.data[0]["id"]

    except Exception:
        logger.exception("Failed to save upload (user_id=%s)", user_id)
        if raise_on_error:
            raise
        return None


def save_prediction(
    user_id: Optional[str],
    upload_id: Optional[str],
    model_used: str,
    result_json: dict,
    confidence: Optional[float],
    raise_on_error: bool = False,
) -> Optional[str]:
    """
    Insert a `predictions` row (upload_id may be None — e.g. root-cause
    predictions have no source file). Returns the new prediction_id, or
    None if user_id is None (anonymous — no-op) or the write failed.

    By default, never raises — errors are logged and swallowed. Pass
    raise_on_error=True (debug-only) to propagate the exception instead.
    """
    if user_id is None:
        return None

    try:
        client = get_supabase()
        prediction_row = (
            client.table("predictions")
            .insert(
                {
                    "upload_id": upload_id,
                    "user_id": user_id,
                    "model_used": model_used,
                    "result_json": result_json,
                    "confidence": confidence,
                }
            )
            .execute()
        )
        return prediction_row.data[0]["id"] if prediction_row.data else None

    except Exception:
        logger.exception(
            "Failed to save prediction (model_used=%s, user_id=%s)", model_used, user_id
        )
        if raise_on_error:
            raise
        return None


def save_upload_and_prediction(
    user_id: Optional[str],
    file_bytes: Optional[bytes],
    original_filename: Optional[str],
    file_type: Optional[str],
    model_context: Optional[str],
    model_used: str,
    result_json: dict,
    confidence: Optional[float],
    raise_on_error: bool = False,
) -> Optional[dict]:
    """
    Convenience wrapper for the common case: one file (or none), one
    prediction. Uploads file_bytes (if given) via save_upload, then inserts
    one predictions row via save_prediction referencing it. Returns
    {"upload_id": ..., "prediction_id": ...} on success, None if user_id is
    None or either step failed to produce an id (partial writes from a
    mid-way failure are left in place rather than rolled back).

    For a batch/multi-model flow where the SAME file is used for more than
    one prediction, call save_upload once and save_prediction per model
    instead of this — using save_upload_and_prediction per model would
    upload and insert the same file multiple times.
    """
    if user_id is None:
        return None

    upload_id = None
    if file_bytes is not None:
        upload_id = save_upload(
            user_id, file_bytes, original_filename, file_type, model_context, raise_on_error
        )

    prediction_id = save_prediction(
        user_id, upload_id, model_used, result_json, confidence, raise_on_error
    )

    return {"upload_id": upload_id, "prediction_id": prediction_id}

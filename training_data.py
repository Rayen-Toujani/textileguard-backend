"""
Part 2 — silently persists predictions (and their source file, if any) for
future model retraining, tied to whichever user happened to be logged in.

Prediction endpoints stay PUBLIC: this module is a best-effort side effect
that must never affect the response the caller is waiting on.
"""

import logging
import mimetypes
from typing import Optional
from uuid import uuid4

from supabase_client import get_supabase

logger = logging.getLogger(__name__)

UPLOADS_BUCKET = "uploads"


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
    Anonymous callers (user_id is None) are a no-op — nothing to save.

    Otherwise: if file_bytes is given, upload it to the "uploads" bucket at
    {user_id}/{uuid}.{ext} and insert an `uploads` row; then always insert a
    `predictions` row (upload_id is None when there was no file). Returns
    {"upload_id": ..., "prediction_id": ...} on success.

    By default, never raises — errors are logged and swallowed so a
    training-data write failure can't break the actual prediction response.
    Pass raise_on_error=True (debug-only) to propagate the exception instead.
    """
    if user_id is None:
        return None

    try:
        client = get_supabase()
        upload_id = None

        if file_bytes is not None:
            extension = ""
            if original_filename and "." in original_filename:
                extension = original_filename.rsplit(".", 1)[1].lower()
            storage_path = (
                f"{user_id}/{uuid4()}.{extension}" if extension else f"{user_id}/{uuid4()}"
            )
            content_type = mimetypes.guess_type(original_filename or "")[0] or "application/octet-stream"

            client.storage.from_(UPLOADS_BUCKET).upload(
                storage_path,
                file_bytes,
                {"content-type": content_type},
            )

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
            upload_id = upload_row.data[0]["id"]

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

        return {
            "upload_id": upload_id,
            "prediction_id": prediction_row.data[0]["id"] if prediction_row.data else None,
        }

    except Exception:
        logger.exception(
            "Failed to save training data (model_used=%s, user_id=%s)", model_used, user_id
        )
        if raise_on_error:
            raise
        return None

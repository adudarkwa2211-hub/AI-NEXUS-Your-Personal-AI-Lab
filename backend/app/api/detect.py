from __future__ import annotations

import base64
import logging
from io import BytesIO
from threading import Lock

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.detector.yolo import ObjectDetector

logger = logging.getLogger(__name__)
router = APIRouter()
_model_lock = Lock()


def get_detector(request: Request) -> ObjectDetector:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        with _model_lock:
            detector = getattr(request.app.state, "detector", None)
            if detector is None:
                detector = ObjectDetector(get_settings().detector_model)
                request.app.state.detector = detector
    return detector


@router.post("/detect")
async def detect(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = Form(default=None, ge=0.01, le=1.0),
) -> dict:
    settings = get_settings()
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image must be at most {settings.max_upload_mb} MB")
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image file") from None

    try:
        result = get_detector(request).predict(
            image, settings.detector_confidence if confidence is None else confidence
        )
    except Exception:
        logger.exception("Object detection failed")
        raise HTTPException(status_code=500, detail="Object detection failed. Check server logs.") from None

    image_bytes = result.pop("image_base64")
    result["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
    return result

from __future__ import annotations

import logging
import time
from io import BytesIO
from threading import Lock

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.classifier.resnet import ImageClassifier
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
_model_lock = Lock()


def get_classifier(request: Request) -> ImageClassifier:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        with _model_lock:
            classifier = getattr(request.app.state, "classifier", None)
            if classifier is None:
                settings = get_settings()
                classifier = ImageClassifier(
                    settings.classifier_model_path,
                    min_confidence=settings.classifier_min_confidence,
                )
                request.app.state.classifier = classifier
    return classifier


@router.post("/classify")
async def classify(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Form(default=5, ge=1, le=5),
) -> dict:
    settings = get_settings()
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image must be at most {settings.max_upload_mb} MB")
    try:
        with Image.open(BytesIO(content)) as uploaded:
            uploaded.load()
            image = uploaded.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Upload a valid image file") from None

    try:
        classifier = get_classifier(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        logger.exception("Could not load flower classifier")
        raise HTTPException(status_code=503, detail="Flower classifier is not available. Check server logs.") from None

    started_at = time.perf_counter()
    try:
        result = classifier.predict(image, top_k=top_k)
    except Exception:
        logger.exception("Flower classification failed")
        raise HTTPException(status_code=500, detail="Flower classification failed. Check server logs.") from None
    result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
    return result

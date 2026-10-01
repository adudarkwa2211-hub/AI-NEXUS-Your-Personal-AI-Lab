import io
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from PIL import Image

from app.core.config import get_settings
from app.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Retrieval"])

# Singleton lưu instance của RetrievalService để tránh load lại CLIP model nhiều lần
_service_instance: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    global _service_instance
    if _service_instance is None:
        settings = get_settings()
        _service_instance = RetrievalService(settings)
    return _service_instance


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Từ khóa/văn bản tìm kiếm")
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultItem(BaseModel):
    id: int
    filename: str
    relative_url: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


@router.post("/retrieval/text", response_model=SearchResponse)
async def search_by_text(payload: TextSearchRequest):
    service = get_retrieval_service()
    try:
        results = service.search_by_text(payload.query, top_k=payload.top_k)
        return {"results": results}
    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tìm kiếm ảnh bằng văn bản: {str(e)}"
        )


@router.post("/retrieval/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=5)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File tải lên phải là hình ảnh"
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        service = get_retrieval_service()
        results = service.search_by_image(image, top_k=int(top_k))
        return {"results": results}
    except Exception as e:
        logger.error(f"Image search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi tìm kiếm ảnh bằng hình ảnh: {str(e)}"
        )
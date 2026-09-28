import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.schemas import ChatRequest, ChatResponse, HealthResponse
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not loaded yet")
    return pipeline


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> ChatResponse:
    try:
        return pipeline.answer(payload.message)
    except FileNotFoundError as exc:
        logger.exception("Knowledge index missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        logger.exception("Chat generation failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer. Check server logs.",
        ) from None


@router.get("/health", response_model=HealthResponse)
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_model=pipeline.settings.embedding_model,
        llm_model=pipeline.settings.llm_model,
        index_loaded=pipeline.store.index is not None,
        index_size=pipeline.store.size,
    )

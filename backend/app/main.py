from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.classify import router as classify_router
from app.api.detect import router as detect_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.pipeline import RAGPipeline


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            app.state.pipeline = RAGPipeline.load(settings)
        yield

    app = FastAPI(
        title="RAG Chatbot",
        description="Multilingual MiniLM + FAISS + Qwen2.5 campus assistant",
        lifespan=lifespan,
    )
    if pipeline is not None:
        app.state.pipeline = pipeline
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router, prefix="/api")
    app.include_router(classify_router, prefix="/api")
    app.include_router(detect_router, prefix="/api")
    return app


app = create_app()

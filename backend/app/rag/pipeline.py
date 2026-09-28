from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.core.schemas import ChatResponse
from app.rag.embeddings import MiniLMEmbedder
from app.rag.llm import QwenGenerator
from app.rag.prompt import build_rag_prompt
from app.rag.retriever import Retriever, load_faiss_store
from app.rag.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedder: MiniLMEmbedder,
        store: FaissVectorStore,
        generator: QwenGenerator,
        settings: Settings,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.settings = settings
        self.retriever = Retriever(
            embedder=embedder,
            store=store,
            top_k=settings.top_k,
            score_threshold=settings.score_threshold,
        )

    @classmethod
    def load(cls, settings: Settings | None = None) -> RAGPipeline:
        cfg = settings or get_settings()
        logger.info("Loading RAG pipeline (models are loaded once, not per request)")
        embedder = MiniLMEmbedder(cfg.embedding_model)
        store = load_faiss_store(cfg.index_path)
        generator = QwenGenerator(cfg.llm_model, cfg.max_new_tokens)
        return cls(embedder=embedder, store=store, generator=generator, settings=cfg)

    def answer(self, question: str) -> ChatResponse:
        sources = self.retriever.retrieve(question)
        prompt = build_rag_prompt(question, sources)
        logger.info("Calling Qwen2.5 with %s source chunks", len(sources))
        answer = self.generator.generate(prompt)
        if not answer:
            answer = "I could not generate an answer. Please try again."
        return ChatResponse(answer=answer, sources=sources)

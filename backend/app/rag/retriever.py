import logging
from pathlib import Path

from app.core.schemas import SourceChunk
from app.rag.embeddings import MiniLMEmbedder
from app.rag.vectorstore import FaissVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        embedder: MiniLMEmbedder,
        store: FaissVectorStore,
        top_k: int,
        score_threshold: float,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, question: str) -> list[SourceChunk]:
        logger.info("Retrieving context for question (%s chars)", len(question))
        query_vec = self.embedder.encode([question])
        hits = self.store.search(query_vec, self.top_k)[0]
        sources = [
            SourceChunk(
                id=int(hit["id"]),
                source=str(hit["source"]),
                score=float(hit["score"]),
                text=str(hit["text"]),
            )
            for hit in hits
            if float(hit["score"]) >= self.score_threshold
        ]
        logger.info("Kept %s chunks above threshold %.2f", len(sources), self.score_threshold)
        return sources


def load_faiss_store(index_dir: Path) -> FaissVectorStore:
    store = FaissVectorStore()
    store.load(index_dir)
    return store

"""Build the FAISS index from the knowledge-base documents."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.rag.embeddings import MiniLMEmbedder  # noqa: E402
from app.rag.ingest import documents_to_chunks, embed_chunks, load_documents  # noqa: E402
from app.rag.vectorstore import FaissVectorStore  # noqa: E402

logger = logging.getLogger("ingest")


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Reading documents from %s", settings.documents_path)
    documents = load_documents(settings.documents_path)
    if not documents:
        raise SystemExit(f"No .txt/.md files found in {settings.documents_path}")
    metadata = documents_to_chunks(
        documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    logger.info("Prepared %s chunks from %s files", len(metadata), len(documents))
    embedder = MiniLMEmbedder(settings.embedding_model)
    vectors = embed_chunks(embedder, metadata)
    store = FaissVectorStore()
    store.build(vectors, metadata)
    store.save(settings.index_path)
    logger.info("Ingest complete: %s", settings.faiss_index_file)


if __name__ == "__main__":
    main()

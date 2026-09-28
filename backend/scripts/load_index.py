"""Load the FAISS index and print a short status report."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.rag.retriever import load_faiss_store  # noqa: E402

logger = logging.getLogger("load_index")


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    store = load_faiss_store(settings.index_path)
    logger.info("Index file: %s", settings.faiss_index_file)
    logger.info("Vectors: %s", store.size)
    logger.info("Metadata records: %s", len(store.metadata))
    if store.metadata:
        logger.info("First source: %s", store.metadata[0].get("source"))


if __name__ == "__main__":
    main()

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MINILM_DIM = 384


class MiniLMEmbedder:
    """Loads multilingual MiniLM once and encodes texts for FAISS."""

    def __init__(self, model_name: str) -> None:
        if "minilm" not in model_name.lower():
            raise ValueError(
                f"Expected a multilingual MiniLM embedding model, got: {model_name}"
            )
        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        if dim != MINILM_DIM:
            raise ValueError(
                f"MiniLM embeddings must be {MINILM_DIM}-dimensional, got {dim}"
            )
        logger.info("Embedding model ready (dim=%s)", dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

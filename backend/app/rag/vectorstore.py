from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """Build, save, and load a FAISS inner-product index of normalized vectors."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    @property
    def size(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)

    def build(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if len(metadata) != vectors.shape[0]:
            raise ValueError("metadata length must match number of vectors")
        dim = vectors.shape[1]
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(matrix)
        self.metadata = metadata
        logger.info("Built FAISS index with %s vectors (dim=%s)", self.size, dim)

    def save(self, index_dir: Path) -> None:
        if self.index is None:
            raise RuntimeError("No FAISS index to save")
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.json"
        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved FAISS index to %s", index_path)

    def load(self, index_dir: Path) -> None:
        index_path = index_dir / "index.faiss"
        meta_path = index_dir / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found in {index_dir}. Run: python -m scripts.ingest"
            )
        self.index = faiss.read_index(str(index_path))
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.size != len(self.metadata):
            raise ValueError("FAISS index size does not match metadata.json")
        logger.info("Loaded FAISS index (%s vectors) from %s", self.size, index_path)

    def search(self, query_vectors: np.ndarray, top_k: int) -> list[list[dict]]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded")
        queries = np.asarray(query_vectors, dtype="float32")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        faiss.normalize_L2(queries)
        k = min(top_k, self.size)
        if k <= 0:
            return [[] for _ in range(queries.shape[0])]
        scores, ids = self.index.search(queries, k)
        batch: list[list[dict]] = []
        for row_scores, row_ids in zip(scores, ids, strict=True):
            hits = []
            for score, idx in zip(row_scores, row_ids, strict=True):
                if idx < 0:
                    continue
                item = dict(self.metadata[int(idx)])
                item["score"] = float(score)
                hits.append(item)
            batch.append(hits)
        return batch

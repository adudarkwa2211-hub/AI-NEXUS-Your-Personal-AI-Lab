from pathlib import Path

import numpy as np
import pytest

from app.rag.ingest import chunk_text, documents_to_chunks
from app.rag.prompt import build_rag_prompt
from app.rag.retriever import Retriever
from app.rag.vectorstore import FaissVectorStore
from app.core.schemas import SourceChunk


class FakeMiniLM:
    """Deterministic 384-d encoder used so retrieval tests do not download models."""

    dim = 384

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in text.lower().split():
                vectors[i, hash(token) % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def test_chunk_text_respects_overlap():
    text = "abcdefghij" * 10
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert chunks[0][:5] == "abcde"


def test_documents_to_chunks_assigns_ids():
    docs = [("a.txt", "hello world " * 50), ("b.txt", "second document " * 50)]
    meta = documents_to_chunks(docs, chunk_size=40, overlap=5)
    assert meta[0]["id"] == 0
    assert meta[-1]["id"] == len(meta) - 1
    sources = {item["source"] for item in meta}
    assert sources == {"a.txt", "b.txt"}


def test_faiss_roundtrip(tmp_path: Path):
    embedder = FakeMiniLM()
    metadata = [
        {"id": 0, "source": "library.txt", "text": "library closes at nine"},
        {"id": 1, "source": "food.txt", "text": "cafeteria serves pho at noon"},
    ]
    vectors = embedder.encode([item["text"] for item in metadata])
    store = FaissVectorStore()
    store.build(vectors, metadata)
    store.save(tmp_path)
    loaded = FaissVectorStore()
    loaded.load(tmp_path)
    assert loaded.size == 2
    hits = loaded.search(embedder.encode(["when does the library close"]), top_k=1)[0]
    assert hits[0]["source"] == "library.txt"


def test_retriever_filters_by_threshold():
    embedder = FakeMiniLM()
    metadata = [
        {"id": 0, "source": "library.txt", "text": "library hours monday friday"},
        {"id": 1, "source": "unrelated.txt", "text": "zzzz other topic"},
    ]
    store = FaissVectorStore()
    store.build(embedder.encode([m["text"] for m in metadata]), metadata)
    all_hits = Retriever(embedder, store, top_k=2, score_threshold=0.0).retrieve(
        "library hours"
    )
    assert all_hits[0].source == "library.txt"
    high = Retriever(embedder, store, top_k=2, score_threshold=0.999).retrieve(
        "library hours"
    )
    assert all(item.score >= 0.999 for item in high)


def test_retriever_returns_best_match():
    embedder = FakeMiniLM()
    metadata = [
        {"id": 0, "source": "library.txt", "text": "the library is open until 21:00"},
        {"id": 1, "source": "admissions.txt", "text": "applications close on 15 june"},
    ]
    store = FaissVectorStore()
    store.build(embedder.encode([m["text"] for m in metadata]), metadata)
    retriever = Retriever(embedder, store, top_k=2, score_threshold=0.0)
    sources = retriever.retrieve("what time is the library open")
    assert sources
    assert sources[0].source == "library.txt"


def test_rag_prompt_includes_context_and_question():
    sources = [
        SourceChunk(id=0, source="library.txt", score=0.9, text="Open until 21:00")
    ]
    prompt = build_rag_prompt("When is the library open?", sources)
    assert "Open until 21:00" in prompt
    assert "When is the library open?" in prompt
    empty = build_rag_prompt("unknown", [])
    assert "do not know" in empty.lower()

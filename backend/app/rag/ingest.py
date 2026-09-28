from pathlib import Path

from app.rag.embeddings import MiniLMEmbedder


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = end - overlap
    return chunks


def load_documents(documents_dir: Path) -> list[tuple[str, str]]:
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    files = sorted(
        p
        for p in documents_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    )
    docs = []
    for path in files:
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def documents_to_chunks(
    documents: list[tuple[str, str]],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    metadata = []
    chunk_id = 0
    for source, text in documents:
        for chunk in chunk_text(text, chunk_size, overlap):
            metadata.append({"id": chunk_id, "source": source, "text": chunk})
            chunk_id += 1
    return metadata


def embed_chunks(embedder: MiniLMEmbedder, metadata: list[dict]):
    texts = [item["text"] for item in metadata]
    return embedder.encode(texts)

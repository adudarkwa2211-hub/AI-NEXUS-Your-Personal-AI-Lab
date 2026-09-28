from app.core.schemas import SourceChunk

SYSTEM_PROMPT = (
    "You are the Greenfield University campus assistant. "
    "Answer using only the retrieved context. "
    "If the context is not enough, say you do not know. "
    "Reply in the same language as the question. "
    "Be concise and factual."
)


def build_rag_prompt(question: str, sources: list[SourceChunk]) -> str:
    if not sources:
        return (
            "No relevant context was retrieved from the knowledge base.\n\n"
            f"Question: {question}\n"
            "Tell the user you do not know based on the campus documents."
        )
    blocks = []
    for i, source in enumerate(sources, start=1):
        blocks.append(
            f"[{i}] source={source.source} score={source.score:.3f}\n{source.text}"
        )
    context = "\n\n".join(blocks)
    return (
        "Use the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

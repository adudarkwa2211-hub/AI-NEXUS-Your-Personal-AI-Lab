from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.schemas import ChatResponse, SourceChunk
from app.main import create_app
from app.rag.vectorstore import FaissVectorStore


class FakePipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = FaissVectorStore()
        self.store.metadata = [{"id": 0, "source": "library.txt", "text": "open 21:00"}]

    def answer(self, question: str) -> ChatResponse:
        if question == "boom":
            raise RuntimeError("generation exploded")
        return ChatResponse(
            answer=f"Echo: {question}",
            sources=[
                SourceChunk(
                    id=0,
                    source="library.txt",
                    score=0.88,
                    text="The library is open until 21:00.",
                )
            ],
        )


def make_client() -> TestClient:
    app = create_app(pipeline=FakePipeline())
    return TestClient(app)


def test_chat_success():
    client = make_client()
    response = client.post("/api/chat", json={"message": "When is the library open?"})
    assert response.status_code == 200
    body = response.json()
    assert "library" in body["answer"].lower() or body["answer"].startswith("Echo:")
    assert body["sources"][0]["source"] == "library.txt"


def test_chat_rejects_empty_message():
    client = make_client()
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_rejects_missing_message():
    client = make_client()
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_chat_handles_generation_error():
    client = make_client()
    response = client.post("/api/chat", json={"message": "boom"})
    assert response.status_code == 500
    assert "Failed to generate" in response.json()["detail"]


def test_health_ok():
    client = make_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "minilm" in body["embedding_model"].lower()
    assert "qwen2.5" in body["llm_model"].lower()

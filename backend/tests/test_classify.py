from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


class FakeClassifier:
    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        assert image.mode == "RGB"
        predictions = [
            {"label": "sunflowers", "score": 0.82},
            {"label": "daisy", "score": 0.08},
            {"label": "tulips", "score": 0.05},
            {"label": "roses", "score": 0.03},
            {"label": "dandelion", "score": 0.02},
        ][:top_k]
        return {"predictions": predictions, "confident": predictions[0]["score"] >= 0.5}


def make_client() -> TestClient:
    app = create_app(pipeline=object())
    app.state.classifier = FakeClassifier()
    return TestClient(app)


def png_bytes() -> bytes:
    image = BytesIO()
    Image.new("RGB", (32, 32), color="yellow").save(image, format="PNG")
    return image.getvalue()


def test_classify_returns_top_k_predictions_and_latency():
    response = make_client().post(
        "/api/classify",
        files={"file": ("flower.png", png_bytes(), "image/png")},
        data={"top_k": "3"},
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["label"] for item in body["predictions"]] == ["sunflowers", "daisy", "tulips"]
    assert body["confident"] is True
    assert body["latency_ms"] >= 0


def test_classify_rejects_non_image():
    response = make_client().post(
        "/api/classify",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a valid image file"


def test_classify_rejects_upload_over_limit():
    response = make_client().post(
        "/api/classify",
        files={"file": ("large.png", b"x" * (8 * 1024 * 1024 + 1), "image/png")},
    )

    assert response.status_code == 413


def test_classify_marks_low_confidence_result():
    client = make_client()
    client.app.state.classifier.predict = lambda image, top_k=5: {
        "predictions": [{"label": "roses", "score": 0.31}],
        "confident": False,
    }
    response = client.post(
        "/api/classify",
        files={"file": ("flower.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["confident"] is False

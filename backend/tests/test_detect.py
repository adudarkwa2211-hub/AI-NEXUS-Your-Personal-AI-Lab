import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app


class FakeDetector:
    def predict(self, image: Image.Image, confidence: float) -> dict:
        output = BytesIO()
        image.save(output, format="JPEG")
        return {
            "model": "yolo11n.pt",
            "image_base64": output.getvalue(),
            "detections": [
                {
                    "label": "person",
                    "class_id": 0,
                    "confidence": confidence,
                    "box_xyxy": [1.0, 2.0, 30.0, 40.0],
                }
            ],
        }


def make_client() -> TestClient:
    app = create_app(pipeline=object())
    app.state.detector = FakeDetector()
    return TestClient(app)


def test_detect_image_returns_detections_and_annotated_image():
    image = BytesIO()
    Image.new("RGB", (32, 32), color="white").save(image, format="PNG")
    response = make_client().post(
        "/api/detect",
        files={"file": ("sample.png", image.getvalue(), "image/png")},
        data={"confidence": "0.4"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "yolo11n.pt"
    assert body["detections"][0]["label"] == "person"
    assert body["detections"][0]["confidence"] == 0.4
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")
    base64.b64decode(body["annotated_image"].split(",", 1)[1], validate=True)


def test_detect_rejects_non_image_upload():
    response = make_client().post(
        "/api/detect",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a valid image file"


def test_detect_rejects_out_of_range_confidence():
    image = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(image, format="PNG")
    response = make_client().post(
        "/api/detect",
        files={"file": ("sample.png", image.getvalue(), "image/png")},
        data={"confidence": "1.5"},
    )

    assert response.status_code == 422

from __future__ import annotations

from io import BytesIO

from PIL import Image


class ObjectDetector:
    """Thin inference wrapper around the Ultralytics COCO pretrained model."""

    def __init__(self, weights: str = "yolo11n.pt") -> None:
        # Import lazily so the chatbot remains usable until detection is requested.
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.weights = weights

    def predict(self, image: Image.Image, confidence: float = 0.25) -> dict:
        result = self.model.predict(image.convert("RGB"), conf=confidence, verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "label": result.names[class_id],
                        "class_id": class_id,
                        "confidence": round(float(box.conf.item()), 4),
                        "box_xyxy": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    }
                )

        annotated = Image.fromarray(result.plot()[..., ::-1])
        output = BytesIO()
        annotated.save(output, format="JPEG", quality=88)
        return {
            "model": self.weights,
            "image_base64": output.getvalue(),
            "detections": detections,
        }

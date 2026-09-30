from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ImageClassifier:
    def __init__(
        self,
        model_dir: Path,
        min_confidence: float = 0.5,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        weights_path = self.model_dir / "model.pt"
        classes_path = self.model_dir / "classes.json"
        if not weights_path.is_file() or not classes_path.is_file():
            raise FileNotFoundError(
                f"Flower classifier artifacts are missing from {self.model_dir}; "
                "run scripts/train_flower_classifier.py first"
            )

        self.classes = json.loads(classes_path.read_text(encoding="utf-8"))
        if not isinstance(self.classes, list) or len(self.classes) < 2:
            raise ValueError(f"Invalid class list in {classes_path}")
        self.min_confidence = min_confidence
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_resnet18(len(self.classes))
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.transform = evaluation_transform()

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = self.model(tensor).softmax(dim=-1)[0]
        scores, indices = probabilities.topk(min(top_k, len(self.classes)))
        predictions = [
            {"label": self.classes[index], "score": round(float(score), 4)}
            for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
        ]
        return {
            "predictions": predictions,
            "confident": bool(predictions and predictions[0]["score"] >= self.min_confidence),
        }

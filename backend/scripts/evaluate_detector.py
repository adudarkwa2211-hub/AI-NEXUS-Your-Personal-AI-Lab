"""Evaluate pretrained YOLO11n on Ultralytics COCO128 and save mAP metrics."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.detector.yolo import ObjectDetector


def main() -> None:
    settings = get_settings()
    detector = ObjectDetector(settings.detector_model)
    result = detector.model.val(data="coco128.yaml", imgsz=640, verbose=False)
    metrics = {
        "model": settings.detector_model,
        "dataset": "COCO128",
        "split": "val",
        "imagesize": 640,
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    output = settings.index_path.parent / "metrics" / "detector_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()

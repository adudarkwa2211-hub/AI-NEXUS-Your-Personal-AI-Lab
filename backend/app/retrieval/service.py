import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from app.core.config import Settings

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model '{settings.clip_model}' on {self.device}...")

        self.processor = CLIPProcessor.from_pretrained(settings.clip_model)
        self.model = CLIPModel.from_pretrained(settings.clip_model).to(self.device)
        self.model.eval()

        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

        # Tự động build hoặc load index khi khởi tạo
        self.load_or_build_index()

    def _extract_image_features(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # Normalize vector về độ dài 1 để dùng Inner Product làm Cosine Similarity
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def _extract_text_features(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def build_index(self):
        """Quét tất cả ảnh trong thư mục images và index vào FAISS"""
        images_dir = self.settings.retrieval_images_path
        index_dir = self.settings.retrieval_index_path
        index_dir.mkdir(parents=True, exist_ok=True)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in images_dir.glob("*") if f.suffix.lower() in valid_extensions]

        if not image_files:
            logger.warning(f"No images found in {images_dir}. Creating empty index.")
            dim = 512  # CLIP vit-base-patch32 embedding size
            self.index = faiss.IndexFlatIP(dim)
            self.metadata = []
            return

        features_list = []
        self.metadata = []

        logger.info(f"Extracting features for {len(image_files)} images...")
        for idx, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path).convert("RGB")
                feat = self._extract_image_features(img)
                features_list.append(feat[0])
                self.metadata.append({
                    "id": idx,
                    "filename": img_path.name,
                    "relative_url": f"/retrieval-images/{img_path.name}"
                })
            except Exception as e:
                logger.error(f"Error processing image {img_path.name}: {e}")

        if features_list:
            matrix = np.array(features_list).astype(np.float32)
            dim = matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(matrix)

            # Lưu index và metadata ra đĩa
            faiss.write_index(self.index, str(index_dir / "clip.index"))
            with open(index_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully indexed {len(self.metadata)} images.")

    def load_or_build_index(self):
        index_file = self.settings.retrieval_index_path / "clip.index"
        meta_file = self.settings.retrieval_index_path / "metadata.json"

        if index_file.exists() and meta_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded FAISS index with {len(self.metadata)} entries.")
            except Exception as e:
                logger.warning(f"Failed to load existing index ({e}), rebuilding...")
                self.build_index()
        else:
            self.build_index()

    def search_by_text(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            return []

        text_feat = self._extract_text_features(text)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(text_feat, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = round(float(score), 4)
                results.append(item)
        return results

    def search_by_image(self, image: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.index or self.index.ntotal == 0:
            return []

        img_feat = self._extract_image_features(image)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(img_feat, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = self.metadata[idx].copy()
                item["score"] = round(float(score), 4)
                results.append(item)
        return results
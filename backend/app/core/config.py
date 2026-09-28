from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    documents_dir: Path = Path("backend/data/documents")
    index_dir: Path = Path("backend/data/indexes")
    top_k: int = Field(default=4, ge=1, le=20)
    max_new_tokens: int = Field(default=256, ge=16, le=2048)
    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)
    score_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    @property
    def documents_path(self) -> Path:
        path = self.documents_dir
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def index_path(self) -> Path:
        path = self.index_dir
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def faiss_index_file(self) -> Path:
        return self.index_path / "index.faiss"

    @property
    def metadata_file(self) -> Path:
        return self.index_path / "metadata.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()

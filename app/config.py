from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    llm_temperature: float = 0
    llm_num_ctx: int = 8192
    llm_num_predict: int = 2048
    llm_timeout_sec: int = 600
    llm_max_retries: int = 2

    ocr_languages: list[str] = ["en"]
    pdf_dpi: int = 250
    max_file_mb: int = 10
    max_pages: int = 10

    hints_enabled: bool = True
    fuzzy_match_min: int = 90
    vector_top_k: int = 3
    vector_sim_threshold: float = 0.75
    min_label_count: int = 1

    prompt_version: str = "v1"
    api_key: str = "change-me"
    data_dir: Path = Path("data")

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def seed_path(self) -> Path:
        return self.data_dir / "seed_labels.json"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_dir, self.outputs_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

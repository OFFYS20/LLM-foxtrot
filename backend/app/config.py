"""Application settings.

Every value is overridable through the environment (prefix ``FOXTROT_``) or a
``.env`` file, so the same image runs in demo mode on a laptop and in a real
GPU workstation without code changes.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

InferenceEngine = Literal["demo", "transformers", "llamacpp", "vllm", "ollama"]
HardwareProviderName = Literal["auto", "nvml", "simulated"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FOXTROT_",
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- identity ----------------------------------------------------------
    app_name: str = "Foxtrot"
    version: str = "0.1.0"

    # --- server ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- storage -----------------------------------------------------------
    database_url: str = "sqlite:///./data/foxtrot.db"
    data_dir: Path = Path("./data")
    max_upload_mb: int = 512

    # --- modes -------------------------------------------------------------
    demo_mode: bool = True
    seed_demo_data: bool = True

    # --- engines -----------------------------------------------------------
    inference_engine: InferenceEngine = "demo"
    hardware_provider: HardwareProviderName = "auto"
    ollama_base_url: str = "http://localhost:11434"

    # --- telemetry ---------------------------------------------------------
    hardware_sample_interval_s: float = Field(default=2.0, ge=0.25, le=60.0)
    hardware_history_points: int = Field(default=360, ge=30, le=5000)
    log_ring_buffer_size: int = Field(default=2000, ge=100, le=100_000)

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, value: Path) -> Path:
        return Path(value).expanduser()

    # --- derived -----------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def checkpoints_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.models_dir,
            self.datasets_dir,
            self.checkpoints_dir,
            self.uploads_dir,
            self.exports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

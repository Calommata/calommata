import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EmbeddingConfig:
    provider: Literal["huggingface", "ollama"] = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "huggingface")
    )  # type: ignore
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    normalize: bool = field(
        default_factory=lambda: os.getenv("EMBEDDING_NORMALIZE", "true").lower()
        == "true"
    )

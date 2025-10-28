import os
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """임베딩 모델 설정"""

    provider: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "huggingface"),
        description="임베딩 제공자 (ollama/huggingface)",
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        description="임베딩 모델 이름",
    )
    # Ollama 설정
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Ollama 서버 URL",
    )
    # HuggingFace 설정
    device: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"),
        description="실행 디바이스 (cpu/cuda)",
    )
    normalize: bool = Field(default=True, description="임베딩 정규화 여부")

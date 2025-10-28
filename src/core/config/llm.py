import os
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM 설정"""

    api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
        description="Google Gemini API 키",
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.0-flash"),
        description="LLM 모델 이름",
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")),
        ge=0.0,
        le=1.0,
        description="LLM 온도",
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")),
        ge=1,
        description="최대 토큰 수",
    )

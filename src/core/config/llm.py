import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.0-flash")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", 0.1))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", 4096))
    )

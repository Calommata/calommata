import os
from typing import Any

from pydantic import BaseModel, Field

from src.core.config.neo4j import Neo4jConfig
from src.core.config.embedding import EmbeddingConfig
from src.core.config.retriever import RetrieverConfig
from src.core.config.llm import LLMConfig


class CoreConfig(BaseModel):
    """Core 패키지 전체 설정"""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    project_name: str = Field(
        default_factory=lambda: os.getenv("PROJECT_NAME", "code-analyzer"),
        description="프로젝트 이름",
    )

    @classmethod
    def from_env(cls) -> "CoreConfig":
        """환경 변수에서 설정 로드"""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return self.model_dump()

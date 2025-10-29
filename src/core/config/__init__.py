import os
from dataclasses import dataclass, field
from typing import Any

from src.core.config.neo4j import Neo4jConfig
from src.core.config.embedding import EmbeddingConfig
from src.core.config.retriever import RetrieverConfig
from src.core.config.llm import LLMConfig


@dataclass
class CoreConfig:
    neo4j: Neo4jConfig = field(default_factory=lambda: Neo4jConfig())
    embedding: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig())
    retriever: RetrieverConfig = field(default_factory=lambda: RetrieverConfig())
    llm: LLMConfig = field(default_factory=lambda: LLMConfig())
    project_name: str = field(
        default_factory=lambda: os.getenv("PROJECT_NAME", "example-project")
    )

    @staticmethod
    def from_env() -> "CoreConfig":
        """환경 변수에서 설정 로드"""
        return CoreConfig()

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return self.__dict__

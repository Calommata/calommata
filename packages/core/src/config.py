"""설정 관리 모듈"""

import os
from typing import Any
from pydantic import BaseModel, Field


class Neo4jConfig(BaseModel):
    """Neo4j 연결 설정"""

    uri: str = Field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        description="Neo4j URI",
    )
    user: str = Field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"),
        description="Neo4j 사용자명",
    )
    password: str = Field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"),
        description="Neo4j 비밀번호",
    )
    batch_size: int = Field(default=500, description="배치 처리 크기")


class EmbeddingConfig(BaseModel):
    """임베딩 모델 설정"""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face 모델 이름",
    )
    device: str = Field(default="cpu", description="실행 디바이스 (cpu/cuda)")
    normalize: bool = Field(default=True, description="임베딩 정규화 여부")


class RetrieverConfig(BaseModel):
    """리트리버 설정"""

    similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="유사도 임계값"
    )
    max_results: int = Field(default=5, ge=1, description="최대 검색 결과 수")
    context_depth: int = Field(default=2, ge=1, description="그래프 탐색 깊이")


class LLMConfig(BaseModel):
    """LLM 설정"""

    api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
        description="Google Gemini API 키",
    )
    model_name: str = Field(default="gemini-1.5-flash", description="LLM 모델 이름")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="LLM 온도")
    max_tokens: int = Field(default=2048, ge=1, description="최대 토큰 수")


class CoreConfig(BaseModel):
    """Core 패키지 전체 설정"""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    project_name: str = Field(default="code-analyzer", description="프로젝트 이름")

    @classmethod
    def from_env(cls) -> "CoreConfig":
        """환경 변수에서 설정 로드"""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return self.model_dump()

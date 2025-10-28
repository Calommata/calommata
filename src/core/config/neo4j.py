import os
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

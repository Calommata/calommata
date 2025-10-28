from pydantic import BaseModel, Field


class RetrieverConfig(BaseModel):
    """리트리버 설정"""

    similarity_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="유사도 임계값"
    )
    max_results: int = Field(default=5, ge=1, description="최대 검색 결과 수")
    context_depth: int = Field(default=2, ge=1, description="그래프 탐색 깊이")

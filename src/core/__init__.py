"""Core 패키지 - 코드 분석 및 GraphRAG 핵심 기능

이 패키지는 다음 기능들을 제공합니다:
- 로컬 LLM 기반 코드 임베딩 (Ollama)
- Gemini 2.0 Flash를 활용한 상용 LLM 지원
- Neo4j GraphRAG 구현
- AST 기반 코드 관계 분석
- LangChain & LangGraph 통합

Python 3.13 호환성:
- 향상된 타입 힌팅 활용
- 새로운 에러 처리 패턴 적용
- 모듈식 설계로 유지보수성 향상
"""

from .agent import CodeRAGAgent, AgentState
from .config import CoreConfig, EmbeddingConfig, LLMConfig, Neo4jConfig, RetrieverConfig
from .embedder import CodeEmbedder
from .factory import create_from_config, create_agent_only
from .graph_service import GraphService
from .retriever import CodeRetriever, CodeSearchResult

__version__ = "0.2.0"

__all__ = [
    # Main classes
    "CodeRAGAgent",
    "CodeEmbedder",
    "CodeRetriever",
    "GraphService",
    # Configuration
    "CoreConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "Neo4jConfig",
    "RetrieverConfig",
    # Data models
    "AgentState",
    "CodeSearchResult",
    # Factory functions
    "create_from_config",
    "create_agent_only",
]

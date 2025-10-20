"""코드 분석 및 GraphRAG 서비스를 위한 Core 패키지

LangChain과 LangGraph를 사용한 AI Agent로 Neo4j 그래프에서
코드를 검색하고 연관된 코드를 RAG 방식으로 탐색합니다.
"""

from .agent import CodeRAGAgent
from .embedder import CodeEmbedder
from .retriever import CodeRetriever, CodeSearchResult
from .graph_service import GraphService
from .config import CoreConfig, Neo4jConfig, EmbeddingConfig, RetrieverConfig, LLMConfig
from .factory import create_from_config, create_agent_only

__all__ = [
    "CodeRAGAgent",
    "CodeEmbedder",
    "CodeRetriever",
    "CodeSearchResult",
    "GraphService",
    "CoreConfig",
    "Neo4jConfig",
    "EmbeddingConfig",
    "RetrieverConfig",
    "LLMConfig",
    "create_from_config",
    "create_agent_only",
]

__version__ = "0.2.0"

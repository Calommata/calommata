"""
Core 패키지
Neo4j, 임베딩, GraphRAG 기능을 제공
"""

from .src.embedding_service import EmbeddingService
from .src.code_vectorizer import CodeVectorizer
from .src.graph_rag import GraphRAGService, RAGConfig

# Neo4jPersistence는 Graph 패키지에서 import
try:
    from graph.src.persistence import Neo4jPersistence
except ImportError:
    Neo4jPersistence = None

__version__ = "0.1.0"
__all__ = [
    "EmbeddingService",
    "CodeVectorizer",
    "GraphRAGService",
    "RAGConfig",
    "Neo4jPersistence",
]

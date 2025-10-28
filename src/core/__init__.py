from .agent import CodeRAGAgent
from .config import CoreConfig, EmbeddingConfig, LLMConfig, Neo4jConfig, RetrieverConfig
from .embedder import CodeEmbedder
from .factory import create_from_config
from .graph_service import CodeGraphService
from .retriever import CodeRetriever, CodeSearchResult
from .state import AgentState
from .constants import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_RELATED_NODES_LIMIT,
    DEFAULT_CONTEXT_DEPTH,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_MAX_RESULTS,
    DEFAULT_NEO4J_BATCH_SIZE,
)
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .context_optimizer import ContextOptimizer

__version__ = "0.2.0"

__all__ = [
    "CodeRAGAgent",
    "CodeEmbedder",
    "CodeRetriever",
    "CodeGraphService",
    "CoreConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "Neo4jConfig",
    "RetrieverConfig",
    "AgentState",
    "CodeSearchResult",
    "create_from_config",
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_RELATED_NODES_LIMIT",
    "DEFAULT_CONTEXT_DEPTH",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_NEO4J_BATCH_SIZE",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "ContextOptimizer",
]

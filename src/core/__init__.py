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
from .project_analyzer import ProjectAnalyzer
from .graph_embedder import GraphEmbedder
from .graph_statistics import GraphStatistics
from .query_understanding_node import QueryUnderstandingNode
from .code_retrieval_node import CodeRetrievalNode
from .context_building_node import ContextBuildingNode
from .answer_generation_node import AnswerGenerationNode

__version__ = "0.2.0"

__all__ = [
    # Main classes
    "CodeRAGAgent",
    "CodeEmbedder",
    "CodeRetriever",
    "CodeGraphService",
    # Config
    "CoreConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "Neo4jConfig",
    "RetrieverConfig",
    # State
    "AgentState",
    "CodeSearchResult",
    # Graph Service components
    "ProjectAnalyzer",
    "GraphEmbedder",
    "GraphStatistics",
    # Agent Node components
    "QueryUnderstandingNode",
    "CodeRetrievalNode",
    "ContextBuildingNode",
    "AnswerGenerationNode",
    # Utility
    "ContextOptimizer",
    # Constants
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_RELATED_NODES_LIMIT",
    "DEFAULT_CONTEXT_DEPTH",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_NEO4J_BATCH_SIZE",
    # Prompts
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    # Factory
    "create_from_config",
]

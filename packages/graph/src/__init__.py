"""
Graph package - Code structure modeling and Neo4j integration

This package provides:
- Data models for code graphs (CodeNode, CodeRelation, CodeGraph)
- Adapter pattern to convert Parser output to Graph models
- Utilities for validation, export, and analysis
- Neo4j persistence layer for graph storage
- Query management and exception handling
"""

from .adapter import ParserToGraphAdapter
from .exceptions import (
    ConnectionError,
    IndexCreationError,
    InvalidDataError,
    NodeNotFoundError,
    PersistenceError,
    QueryExecutionError,
)
from .models import (
    CodeGraph,
    CodeNode,
    CodeRelation,
    Dependency,
    NodeType,
    RelationType,
)
from .persistence import Neo4jPersistence
from .queries import Neo4jQueries
from .utils import GraphAnalyzer, GraphExporter, GraphValidator

__version__ = "0.3.0"

__all__ = [
    # Models
    "CodeGraph",
    "CodeNode",
    "CodeRelation",
    "Dependency",
    "NodeType",
    "RelationType",
    # Adapter
    "ParserToGraphAdapter",
    # Utilities
    "GraphValidator",
    "GraphExporter",
    "GraphAnalyzer",
    # Persistence
    "Neo4jPersistence",
    "Neo4jQueries",
    # Exceptions
    "PersistenceError",
    "ConnectionError",
    "QueryExecutionError",
    "NodeNotFoundError",
    "InvalidDataError",
    "IndexCreationError",
]

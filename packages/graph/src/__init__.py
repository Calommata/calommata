"""
Graph package - Code structure modeling and Neo4j integration

This package provides:
- Data models for code graphs (CodeNode, CodeRelation, CodeGraph)
- Adapter pattern to convert Parser output to Graph models
- Utilities for validation, export, and analysis
- Neo4j persistence layer for graph storage
"""

from .adapter import ParserToGraphAdapter
from .models import (
    CodeGraph,
    CodeNode,
    CodeRelation,
    Dependency,
    NodeType,
    RelationType,
)
from .persistence import Neo4jPersistence
from .utils import GraphAnalyzer, GraphExporter, GraphValidator

__version__ = "0.2.0"

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
]

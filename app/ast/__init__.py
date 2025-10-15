"""AST parsing and graph building module with language-specific features."""

# Core functionality
from app.ast.graph import CodeGraph
from app.ast.graph_builder import GraphBuilder, build_graph_from_files
from app.ast.parser import (
    detect_language,
    get_parser,
    get_supported_extensions,
    get_supported_languages,
    parse_file,
    parse_source,
)
from app.ast.parser_factory import ParserFactory

# Base models
from app.ast.base_models import (
    BaseNode,
    BaseRelation,
    LanguageType,
    NodeType,
    RelationType,
)

# Language-specific models
from app.ast.models import (
    JavaScriptNode,
    JavaScriptRelation,
    ParsedNode,  # For backward compatibility
    ParsedRelation,  # For backward compatibility
    PythonNode,
    PythonRelation,
    TypeScriptNode,
    TypeScriptRelation,
)

# Language-specific parsers
from app.ast.languages.parsers.javascript_parser import JavaScriptParser
from app.ast.languages.parsers.python_parser import PythonParser
from app.ast.languages.parsers.typescript_parser import TypeScriptParser

__all__ = [
    # Core functionality
    "CodeGraph",
    "GraphBuilder",
    "build_graph_from_files",
    "parse_file",
    "parse_source",
    "detect_language",
    "get_parser",
    "get_supported_extensions",
    "get_supported_languages",
    "ParserFactory",
    # Base types
    "BaseNode",
    "BaseRelation",
    "LanguageType",
    "NodeType",
    "RelationType",
    # Language-specific models
    "PythonNode",
    "PythonRelation",
    "TypeScriptNode",
    "TypeScriptRelation",
    "JavaScriptNode",
    "JavaScriptRelation",
    # Language-specific parsers
    "PythonParser",
    "TypeScriptParser",
    "JavaScriptParser",
    # Backward compatibility
    "ParsedNode",
    "ParsedRelation",
]

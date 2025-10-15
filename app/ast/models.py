"""Re-export main AST data types and models.

This module provides convenient access to the base models and language-specific models.
"""

# Re-export base models
from app.ast.base_models import (
    BaseNode,
    BaseRelation,
    LanguageType,
    NodeType,
    ParsedNode,  # For backward compatibility
    ParsedRelation,  # For backward compatibility
    RelationType,
)

# Re-export language-specific models
from app.ast.languages.javascript_models import JavaScriptNode, JavaScriptRelation
from app.ast.languages.python_models import PythonNode, PythonRelation
from app.ast.languages.typescript_models import TypeScriptNode, TypeScriptRelation

__all__ = [
    # Base types
    "BaseNode",
    "BaseRelation",
    "LanguageType",
    "NodeType",
    "RelationType",
    # Backward compatibility
    "ParsedNode",
    "ParsedRelation",
    # Language-specific models
    "PythonNode",
    "PythonRelation",
    "TypeScriptNode",
    "TypeScriptRelation",
    "JavaScriptNode",
    "JavaScriptRelation",
]

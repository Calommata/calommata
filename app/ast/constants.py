"""Optimized AST parser constants with lazy loading.

This module provides convenient access to constants from all supported languages
with lazy loading to avoid import performance issues.
"""

from functools import lru_cache
from typing import cast

LanguageConstants = dict[str, set[str] | dict[str, str]]
CombinedConstants = dict[str, set[str] | dict[str, str]]


@lru_cache(maxsize=1)
def _get_python_constants() -> LanguageConstants:
    """Lazy load Python constants."""
    from app.ast.languages.python.python_constants import (
        PYTHON_CALL_TYPES,
        PYTHON_DEFINITION_TYPES,
        PYTHON_IMPORT_TYPES,
        PYTHON_NODE_TYPE_MAPPING,
        PYTHON_PARENT_DEFINITION_TYPES,
        PYTHON_IDENTIFIER_TYPES,
        PYTHON_COMPLEX_NAME_TYPES,
    )

    return {
        "call_types": PYTHON_CALL_TYPES,
        "definition_types": PYTHON_DEFINITION_TYPES,
        "import_types": PYTHON_IMPORT_TYPES,
        "node_type_mapping": PYTHON_NODE_TYPE_MAPPING,
        "parent_definition_types": PYTHON_PARENT_DEFINITION_TYPES,
        "identifier_types": PYTHON_IDENTIFIER_TYPES,
        "complex_name_types": PYTHON_COMPLEX_NAME_TYPES,
    }


@lru_cache(maxsize=1)
def _get_typescript_constants() -> LanguageConstants:
    """Lazy load TypeScript constants."""
    from app.ast.languages.typescript.typescript_constants import (
        TYPESCRIPT_CALL_TYPES,
        TYPESCRIPT_DEFINITION_TYPES,
        TYPESCRIPT_IMPORT_TYPES,
        TYPESCRIPT_NODE_TYPE_MAPPING,
        TYPESCRIPT_PARENT_DEFINITION_TYPES,
        TYPESCRIPT_IDENTIFIER_TYPES,
        TYPESCRIPT_COMPLEX_NAME_TYPES,
    )

    return {
        "call_types": TYPESCRIPT_CALL_TYPES,
        "definition_types": TYPESCRIPT_DEFINITION_TYPES,
        "import_types": TYPESCRIPT_IMPORT_TYPES,
        "node_type_mapping": TYPESCRIPT_NODE_TYPE_MAPPING,
        "parent_definition_types": TYPESCRIPT_PARENT_DEFINITION_TYPES,
        "identifier_types": TYPESCRIPT_IDENTIFIER_TYPES,
        "complex_name_types": TYPESCRIPT_COMPLEX_NAME_TYPES,
    }


@lru_cache(maxsize=1)
def _get_javascript_constants() -> LanguageConstants:
    """Lazy load JavaScript constants."""
    from app.ast.languages.javascript.javascript_constants import (
        JAVASCRIPT_CALL_TYPES,
        JAVASCRIPT_DEFINITION_TYPES,
        JAVASCRIPT_IMPORT_TYPES,
        JAVASCRIPT_NODE_TYPE_MAPPING,
        JAVASCRIPT_PARENT_DEFINITION_TYPES,
        JAVASCRIPT_IDENTIFIER_TYPES,
        JAVASCRIPT_COMPLEX_NAME_TYPES,
    )

    return {
        "call_types": JAVASCRIPT_CALL_TYPES,
        "definition_types": JAVASCRIPT_DEFINITION_TYPES,
        "import_types": JAVASCRIPT_IMPORT_TYPES,
        "node_type_mapping": JAVASCRIPT_NODE_TYPE_MAPPING,
        "parent_definition_types": JAVASCRIPT_PARENT_DEFINITION_TYPES,
        "identifier_types": JAVASCRIPT_IDENTIFIER_TYPES,
        "complex_name_types": JAVASCRIPT_COMPLEX_NAME_TYPES,
    }


@lru_cache(maxsize=1)
def get_combined_constants() -> CombinedConstants:
    """Get combined constants from all languages."""
    python = _get_python_constants()
    typescript = _get_typescript_constants()
    javascript = _get_javascript_constants()

    # Type cast to ensure proper types for union operations
    return {
        "definition_types": cast(set[str], python["definition_types"])
        | cast(set[str], typescript["definition_types"])
        | cast(set[str], javascript["definition_types"]),
        "call_types": cast(set[str], python["call_types"])
        | cast(set[str], typescript["call_types"])
        | cast(set[str], javascript["call_types"]),
        "import_types": cast(set[str], python["import_types"])
        | cast(set[str], typescript["import_types"])
        | cast(set[str], javascript["import_types"]),
        "node_type_mapping": {
            **cast(dict[str, str], python["node_type_mapping"]),
            **cast(dict[str, str], typescript["node_type_mapping"]),
            **cast(dict[str, str], javascript["node_type_mapping"]),
        },
        "parent_definition_types": cast(set[str], python["parent_definition_types"])
        | cast(set[str], typescript["parent_definition_types"])
        | cast(set[str], javascript["parent_definition_types"]),
        "identifier_types": cast(set[str], python["identifier_types"])
        | cast(set[str], typescript["identifier_types"])
        | cast(set[str], javascript["identifier_types"]),
        "complex_name_types": cast(set[str], python["complex_name_types"])
        | cast(set[str], typescript["complex_name_types"])
        | cast(set[str], javascript["complex_name_types"]),
    }


def get_parent_definition_types():
    """Get combined parent definition types from all languages."""
    return get_combined_constants()["parent_definition_types"]


def get_identifier_types():
    """Get combined identifier types from all languages."""
    return get_combined_constants()["identifier_types"]


def get_complex_name_types():
    """Get combined complex name types from all languages."""
    return get_combined_constants()["complex_name_types"]


# Backward compatibility - lazy initialization
PARENT_DEFINITION_TYPES = get_parent_definition_types()
IDENTIFIER_TYPES = get_identifier_types()
COMPLEX_NAME_TYPES = get_complex_name_types()

"""Python-specific AST models and node types."""

from dataclasses import dataclass, field
from typing import Any, Literal

from app.ast.base.base_models import (
    BaseNode,
    BaseRelation,
    LanguageType,
    NodeType,
    RelationType,
)


# Python-specific node types
PythonNodeType = (
    NodeType
    | Literal["decorator", "async_function", "lambda", "comprehension", "generator"]
)

# Python-specific relation types
PythonRelationType = RelationType | Literal["decorates", "yields", "comprehends"]


@dataclass
class PythonNode(BaseNode):
    """Python-specific AST node with Python language features."""

    language: LanguageType = "python"

    # Python-specific attributes
    is_async: bool = False
    is_generator: bool = False
    has_decorators: bool = False
    decorators: list[str] = field(default_factory=lambda: list[str]())
    is_lambda: bool = False
    is_comprehension: bool = False
    comprehension_type: str | None = None  # list, dict, set, generator

    def get_language_features(self) -> dict[str, Any]:
        """Get Python-specific features."""
        features: dict[str, Any] = {
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "has_decorators": self.has_decorators,
            "decorators": self.decorators,
            "is_lambda": self.is_lambda,
            "is_comprehension": self.is_comprehension,
            "comprehension_type": self.comprehension_type,
        }
        features.update(self.metadata)
        return features

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including Python-specific features."""
        base_dict = super().to_dict()
        base_dict.update(self.get_language_features())
        return base_dict


@dataclass
class PythonRelation(BaseRelation):
    """Python-specific relationship with Python language features."""

    # Python-specific relation metadata
    is_async_call: bool = False
    is_yield: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including Python-specific features."""
        base_dict = super().to_dict()
        base_dict["metadata"].update(
            {
                "is_async_call": self.is_async_call,
                "is_yield": self.is_yield,
            }
        )
        return base_dict


# Helper functions for Python-specific features
def create_python_function_node(
    id: str,
    name: str,
    file_path: str,
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
    source_code: str,
    is_async: bool = False,
    decorators: list[str] | None = None,
    parent_id: str | None = None,
) -> PythonNode:
    """Create a Python function node with Python-specific features."""
    return PythonNode(
        id=id,
        type="function",
        name=name,
        file_path=file_path,
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        parent_id=parent_id,
        is_async=is_async,
        has_decorators=bool(decorators),
        decorators=decorators or [],
    )


def create_python_class_node(
    id: str,
    name: str,
    file_path: str,
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
    source_code: str,
    decorators: list[str] | None = None,
    parent_id: str | None = None,
) -> PythonNode:
    """Create a Python class node with Python-specific features."""
    return PythonNode(
        id=id,
        type="class",
        name=name,
        file_path=file_path,
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        parent_id=parent_id,
        has_decorators=bool(decorators),
        decorators=decorators or [],
    )

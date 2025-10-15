"""JavaScript-specific AST models and node types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from app.ast.base_models import (
    BaseNode,
    BaseRelation,
    LanguageType,
    NodeType,
    RelationType,
)


# JavaScript-specific node types
JavaScriptNodeType = (
    NodeType
    | Literal["arrow_function", "closure", "prototype_method", "iife", "callback"]
)

# JavaScript-specific relation types
JavaScriptRelationType = (
    RelationType | Literal["prototype_chain", "closure_captures", "callback_relation"]
)


@dataclass
class JavaScriptNode(BaseNode):
    """JavaScript-specific AST node with JavaScript language features."""

    language: LanguageType = "javascript"

    # JavaScript-specific attributes
    is_arrow_function: bool = False
    is_anonymous: bool = False
    is_iife: bool = False  # Immediately Invoked Function Expression
    is_callback: bool = False
    is_prototype_method: bool = False
    captures_closure: bool = False
    captured_variables: List[str] = field(default_factory=list)
    is_constructor: bool = False
    is_generator: bool = False

    def get_language_features(self) -> Dict[str, Any]:
        """Get JavaScript-specific features."""
        features = {
            "is_arrow_function": self.is_arrow_function,
            "is_anonymous": self.is_anonymous,
            "is_iife": self.is_iife,
            "is_callback": self.is_callback,
            "is_prototype_method": self.is_prototype_method,
            "captures_closure": self.captures_closure,
            "captured_variables": self.captured_variables,
            "is_constructor": self.is_constructor,
            "is_generator": self.is_generator,
        }
        features.update(self.metadata)
        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including JavaScript-specific features."""
        base_dict = super().to_dict()
        base_dict.update(self.get_language_features())
        return base_dict


@dataclass
class JavaScriptRelation(BaseRelation):
    """JavaScript-specific relationship with JavaScript language features."""

    # JavaScript-specific relation metadata
    is_callback_call: bool = False
    is_prototype_access: bool = False
    closure_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including JavaScript-specific features."""
        base_dict = super().to_dict()
        base_dict["metadata"].update(
            {
                "is_callback_call": self.is_callback_call,
                "is_prototype_access": self.is_prototype_access,
                "closure_depth": self.closure_depth,
            }
        )
        return base_dict


# Helper functions for JavaScript-specific features
def create_javascript_function_node(
    id: str,
    name: str,
    file_path: str,
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
    source_code: str,
    is_arrow_function: bool = False,
    is_anonymous: bool = False,
    is_generator: bool = False,
    captures_closure: bool = False,
    captured_variables: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
) -> JavaScriptNode:
    """Create a JavaScript function node with JavaScript-specific features."""
    return JavaScriptNode(
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
        is_arrow_function=is_arrow_function,
        is_anonymous=is_anonymous,
        is_generator=is_generator,
        captures_closure=captures_closure,
        captured_variables=captured_variables or [],
    )


def create_javascript_class_node(
    id: str,
    name: str,
    file_path: str,
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
    source_code: str,
    is_constructor: bool = False,
    parent_id: Optional[str] = None,
) -> JavaScriptNode:
    """Create a JavaScript class node with JavaScript-specific features."""
    return JavaScriptNode(
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
        is_constructor=is_constructor,
    )

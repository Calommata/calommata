"""TypeScript-specific AST models and node types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from app.ast.base_models import (
    BaseNode,
    BaseRelation,
    LanguageType,
    NodeType,
    RelationType,
)


# TypeScript-specific node types
TypeScriptNodeType = (
    NodeType
    | Literal["generic", "interface_property", "type_parameter", "enum", "namespace"]
)

# TypeScript-specific relation types
TypeScriptRelationType = (
    RelationType
    | Literal[
        "extends_interface", "generic_constraint", "enum_member", "namespace_member"
    ]
)


@dataclass
class TypeScriptNode(BaseNode):
    """TypeScript-specific AST node with TypeScript language features."""

    language: LanguageType = "typescript"

    # TypeScript-specific attributes
    has_type_annotation: bool = False
    type_annotation: Optional[str] = None
    is_generic: bool = False
    generic_parameters: List[str] = field(default_factory=list)
    is_interface_member: bool = False
    is_enum: bool = False
    is_namespace: bool = False
    access_modifier: Optional[str] = None  # public, private, protected
    is_readonly: bool = False
    is_optional: bool = False
    is_abstract: bool = False

    def get_language_features(self) -> Dict[str, Any]:
        """Get TypeScript-specific features."""
        features = {
            "has_type_annotation": self.has_type_annotation,
            "type_annotation": self.type_annotation,
            "is_generic": self.is_generic,
            "generic_parameters": self.generic_parameters,
            "is_interface_member": self.is_interface_member,
            "is_enum": self.is_enum,
            "is_namespace": self.is_namespace,
            "access_modifier": self.access_modifier,
            "is_readonly": self.is_readonly,
            "is_optional": self.is_optional,
            "is_abstract": self.is_abstract,
        }
        features.update(self.metadata)
        return features

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including TypeScript-specific features."""
        base_dict = super().to_dict()
        base_dict.update(self.get_language_features())
        return base_dict


@dataclass
class TypeScriptRelation(BaseRelation):
    """TypeScript-specific relationship with TypeScript language features."""

    # TypeScript-specific relation metadata
    is_type_constraint: bool = False
    generic_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including TypeScript-specific features."""
        base_dict = super().to_dict()
        base_dict["metadata"].update(
            {
                "is_type_constraint": self.is_type_constraint,
                "generic_type": self.generic_type,
            }
        )
        return base_dict


# Helper functions for TypeScript-specific features
def create_typescript_function_node(
    id: str,
    name: str,
    file_path: str,
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
    source_code: str,
    type_annotation: Optional[str] = None,
    generic_parameters: Optional[List[str]] = None,
    access_modifier: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> TypeScriptNode:
    """Create a TypeScript function node with TypeScript-specific features."""
    return TypeScriptNode(
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
        has_type_annotation=bool(type_annotation),
        type_annotation=type_annotation,
        is_generic=bool(generic_parameters),
        generic_parameters=generic_parameters or [],
        access_modifier=access_modifier,
    )


def create_typescript_interface_node(
    id: str,
    name: str,
    file_path: str,
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
    source_code: str,
    generic_parameters: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
) -> TypeScriptNode:
    """Create a TypeScript interface node with TypeScript-specific features."""
    return TypeScriptNode(
        id=id,
        type="interface",
        name=name,
        file_path=file_path,
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        parent_id=parent_id,
        is_generic=bool(generic_parameters),
        generic_parameters=generic_parameters or [],
    )


def create_typescript_class_node(
    id: str,
    name: str,
    file_path: str,
    start_byte: int,
    end_byte: int,
    start_line: int,
    end_line: int,
    source_code: str,
    generic_parameters: Optional[List[str]] = None,
    access_modifier: Optional[str] = None,
    is_abstract: bool = False,
    parent_id: Optional[str] = None,
) -> TypeScriptNode:
    """Create a TypeScript class node with TypeScript-specific features."""
    return TypeScriptNode(
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
        is_generic=bool(generic_parameters),
        generic_parameters=generic_parameters or [],
        access_modifier=access_modifier,
        is_abstract=is_abstract,
    )

"""Base AST data types and models shared across languages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

# Language type definition
LanguageType = Literal["python", "javascript", "typescript"]

# Common node types shared across all languages
NodeType = Literal[
    "function", "class", "import", "method", "interface", "type", "unknown"
]

# Relation types shared across languages
RelationType = Literal["calls", "imports", "inherits", "references", "implements"]


@dataclass
class BaseNode(ABC):
    """Base class for AST nodes shared across all languages.

    Contains only the essential properties that all languages share.
    Language-specific features are handled in subclasses.
    """

    id: str  # Unique identifier (file_path:start_byte:end_byte)
    type: NodeType  # Base node type (function, class, etc.)
    name: str | None  # Identifier name if applicable
    file_path: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    source_code: str
    parent_id: str | None = None
    language: LanguageType = "python"  # Language this node belongs to

    # Language-specific metadata
    metadata: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph storage."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "file_path": self.file_path,
            "byte_range": [self.start_byte, self.end_byte],
            "line_range": [self.start_line, self.end_line],
            "source_code": self.source_code,
            "parent_id": self.parent_id,
            "language": self.language,
            "metadata": self.metadata,
        }

    @abstractmethod
    def get_language_features(self) -> dict[str, Any]:
        """Get language-specific features for this node.

        Returns:
            Dictionary containing language-specific metadata
        """
        pass


@dataclass
class BaseRelation(ABC):
    """Base class for relationships between AST nodes."""

    from_id: str
    to_id: str
    relation_type: RelationType
    metadata: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for graph storage."""
        return {
            "from": self.from_id,
            "to": self.to_id,
            "type": self.relation_type,
            "metadata": self.metadata,
        }

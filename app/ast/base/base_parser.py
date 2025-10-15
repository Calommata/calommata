"""Base AST Parser interface and common functionality."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from app.ast.base.base_models import BaseNode, BaseRelation, LanguageType


class BaseASTParser(ABC):
    """Abstract base class for language-specific AST parsers."""

    def __init__(self, language: LanguageType):
        """Initialize parser for specific language.

        Args:
            language: Programming language to parse
        """
        self.language = language

    @abstractmethod
    def parse(
        self, source_code: str, file_path: str
    ) -> Tuple[list[BaseNode], list[BaseRelation]]:
        """Parse source code and extract nodes and relationships.

        Args:
            source_code: Source code to parse
            file_path: Path to the source file (for ID generation)

        Returns:
            Tuple of (nodes, relations)
        """
        pass

    def parse_file(self, file_path: str) -> Tuple[list[BaseNode], list[BaseRelation]]:
        """Parse a source file and extract graph structure.

        Args:
            file_path: Path to source file

        Returns:
            Tuple of (nodes, relations)
        """
        source_code = Path(file_path).read_text(encoding="utf-8")
        return self.parse(source_code, file_path)

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions for this parser.

        Returns:
            List of file extensions (e.g., ['.py', '.pyi'])
        """
        pass

    @abstractmethod
    def extract_language_specific_features(
        self, node: BaseNode, source_code: str
    ) -> BaseNode:
        """Extract language-specific features from a node.

        Args:
            node: Base node to enhance with language features
            source_code: Original source code for context

        Returns:
            Enhanced node with language-specific features
        """
        pass

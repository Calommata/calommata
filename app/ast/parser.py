"""Main AST Parser interface using language-specific parsers."""

from typing import List, Tuple

from app.ast.base_models import BaseNode, BaseRelation, LanguageType
from app.ast.parser_factory import ParserFactory


def parse_file(file_path: str) -> Tuple[List[BaseNode], List[BaseRelation]]:
    """Parse a source file and extract graph structure with language-specific features.

    Args:
        file_path: Path to source file (language auto-detected from extension)

    Returns:
        Tuple of (nodes, relations) with full language-specific features
    """
    return ParserFactory.parse_file(file_path)


def parse_source(
    source_code: str, file_path: str, language: LanguageType | None = None
) -> Tuple[List[BaseNode], List[BaseRelation]]:
    """Parse source code with language-specific features.

    Args:
        source_code: Source code to parse
        file_path: Path to the source file (for ID generation)
        language: Programming language (auto-detected if None)

    Returns:
        Tuple of (nodes, relations) with full language-specific features
    """
    return ParserFactory.parse_source(source_code, file_path, language)


def get_parser(language: LanguageType):
    """Get a language-specific parser instance.

    Args:
        language: Programming language

    Returns:
        Language-specific parser instance
    """
    return ParserFactory.get_parser(language)


def detect_language(file_path: str) -> LanguageType:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the source file

    Returns:
        Detected language type
    """
    return ParserFactory.detect_language(file_path)


def get_supported_extensions() -> List[str]:
    """Get all supported file extensions.

    Returns:
        List of supported file extensions
    """
    return ParserFactory.get_supported_extensions()


def get_supported_languages() -> List[LanguageType]:
    """Get all supported programming languages.

    Returns:
        List of supported language types
    """
    return ParserFactory.get_supported_languages()

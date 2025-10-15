"""Parser factory for language-specific AST parsers."""

from pathlib import Path
from typing import Dict, List, Tuple

from app.ast.base_models import BaseNode, BaseRelation, LanguageType
from app.ast.base_parser import BaseASTParser
from app.ast.languages.parsers.javascript_parser import JavaScriptParser
from app.ast.languages.parsers.python_parser import PythonParser
from app.ast.languages.parsers.typescript_parser import TypeScriptParser


class ParserFactory:
    """Factory class for creating language-specific parsers."""

    # Mapping of file extensions to language types
    EXTENSION_TO_LANGUAGE: Dict[str, LanguageType] = {
        # Python
        ".py": "python",
        ".pyi": "python",
        ".pyx": "python",
        # JavaScript
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        # TypeScript
        ".ts": "typescript",
        ".tsx": "typescript",
        ".d.ts": "typescript",
    }

    # Parser instances (singleton pattern)
    _parsers: Dict[LanguageType, BaseASTParser] = {}

    @classmethod
    def get_parser(cls, language: LanguageType) -> BaseASTParser:
        """Get parser instance for a specific language.

        Args:
            language: Programming language

        Returns:
            Language-specific parser instance
        """
        if language not in cls._parsers:
            if language == "python":
                cls._parsers[language] = PythonParser()
            elif language == "javascript":
                cls._parsers[language] = JavaScriptParser()
            elif language == "typescript":
                cls._parsers[language] = TypeScriptParser()
            else:
                raise ValueError(f"Unsupported language: {language}")

        return cls._parsers[language]

    @classmethod
    def get_parser_for_file(cls, file_path: str) -> BaseASTParser:
        """Get appropriate parser for a file based on its extension.

        Args:
            file_path: Path to the source file

        Returns:
            Language-specific parser instance

        Raises:
            ValueError: If file extension is not supported
        """
        language = cls.detect_language(file_path)
        return cls.get_parser(language)

    @classmethod
    def detect_language(cls, file_path: str) -> LanguageType:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Detected language type

        Raises:
            ValueError: If file extension is not supported
        """
        path = Path(file_path)

        # Handle special case for .d.ts files
        if path.name.endswith(".d.ts"):
            return "typescript"

        extension = path.suffix.lower()

        if extension not in cls.EXTENSION_TO_LANGUAGE:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported extensions: {list(cls.EXTENSION_TO_LANGUAGE.keys())}"
            )

        return cls.EXTENSION_TO_LANGUAGE[extension]

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get all supported file extensions.

        Returns:
            List of supported file extensions
        """
        return list(cls.EXTENSION_TO_LANGUAGE.keys())

    @classmethod
    def get_supported_languages(cls) -> List[LanguageType]:
        """Get all supported programming languages.

        Returns:
            List of supported language types
        """
        return ["python", "javascript", "typescript"]

    @classmethod
    def parse_file(cls, file_path: str) -> Tuple[List[BaseNode], List[BaseRelation]]:
        """Parse a source file using the appropriate language-specific parser.

        Args:
            file_path: Path to source file

        Returns:
            Tuple of (nodes, relations) with language-specific features
        """
        parser = cls.get_parser_for_file(file_path)
        return parser.parse_file(file_path)

    @classmethod
    def parse_source(
        cls, source_code: str, file_path: str, language: LanguageType | None = None
    ) -> Tuple[List[BaseNode], List[BaseRelation]]:
        """Parse source code using language-specific parser.

        Args:
            source_code: Source code to parse
            file_path: Path to the source file (for ID generation)
            language: Programming language (auto-detected if None)

        Returns:
            Tuple of (nodes, relations) with language-specific features
        """
        if language is None:
            language = cls.detect_language(file_path)

        parser = cls.get_parser(language)
        return parser.parse(source_code, file_path)


# Convenience function for backward compatibility
def parse_file(
    file_path: str, language: LanguageType | None = None
) -> Tuple[List[BaseNode], List[BaseRelation]]:
    """Parse a source file and extract graph structure.

    This is the main entry point for parsing files with language-specific features.

    Args:
        file_path: Path to source file
        language: Programming language (auto-detected if None)

    Returns:
        Tuple of (nodes, relations) with language-specific features
    """
    if language is None:
        return ParserFactory.parse_file(file_path)
    else:
        parser = ParserFactory.get_parser(language)
        return parser.parse_file(file_path)

"""Base parser implementation using tree-sitter"""

import logging

from tree_sitter import Language, Parser, Tree

logger = logging.getLogger(__name__)


class BaseParser:
    """Tree-sitter 기반 파서 래퍼 클래스

    특정 언어의 파서를 관리하고 소스 코드를 파싱하는 기능을 제공합니다.

    Attributes:
        language: Tree-sitter 언어 객체
        parser: Tree-sitter 파서 인스턴스
    """

    def __init__(self, lang: object) -> None:
        """파서 초기화

        Args:
            lang: Tree-sitter 언어 객체 (예: tree_sitter_python.language())

        Raises:
            TypeError: lang이 유효하지 않은 경우
        """
        try:
            self.language = Language(lang)
            self.parser = Parser(self.language)
            logger.debug(f"Parser initialized with language: {lang}")
        except Exception as e:
            logger.error(f"Failed to initialize parser with language: {lang}")
            raise TypeError(f"Invalid language object: {e}") from e

    def parse_code(self, source_code: str) -> Tree:
        """소스 코드를 파싱하여 구문 트리 반환

        Args:
            source_code: 파싱할 Python 소스 코드

        Returns:
            파싱된 구문 트리

        Raises:
            ValueError: source_code가 빈 문자열인 경우
            Exception: 파싱 중 오류 발생 시
        """
        if not source_code:
            logger.warning("Empty source code provided")
            raise ValueError("Source code cannot be empty")

        try:
            tree = self.parser.parse(bytes(source_code, "utf-8"))
            logger.debug(f"Successfully parsed code with {len(source_code)} bytes")
            return tree
        except Exception as e:
            logger.error(f"Failed to parse source code: {e}")
            raise

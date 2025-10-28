import logging
from pathlib import Path

from src.parser.queries import PYTHON_QUERIES

from .ast_extractor import ASTExtractor
from .base_parser import BaseParser
from .code_block import CodeBlock, BlockType
from .file_reader import FileReader
import tree_sitter_python

logger = logging.getLogger(__name__)


class CodeASTAnalyzer:
    """코드 분석기 - Tree-sitter를 사용하여 코드 블록 추출

    디렉토리 또는 파일 내의 Python 코드를 분석하여
    함수, 클래스 등의 코드 블록으로 변환합니다.

    Attributes:
        parser: Tree-sitter 기반 파서
        extractor: AST 추출기
        analyzed_blocks: 분석된 모든 코드 블록들
    """

    def __init__(self) -> None:
        """분석기 초기화"""
        self.parser = BaseParser(tree_sitter_python.language())
        self.extractor = ASTExtractor(self.parser.language, PYTHON_QUERIES)
        self.file_reader = FileReader()
        self.analyzed_blocks: list[CodeBlock] = []

    def analyze_file(self, file_path: str) -> list[CodeBlock]:
        """단일 파일 분석

        Args:
            file_path: 분석할 Python 파일 경로

        Returns:
            추출된 CodeBlock들의 리스트

        Raises:
            FileNotFoundError: 파일이 없는 경우
            IOError: 파일 읽기 실패 시
        """
        logger.info(f"Analyzing file: {file_path}")

        try:
            source_code = self.file_reader.read_file(file_path)
            tree = self.parser.parse_code(source_code)
            blocks = self.extractor.extract_blocks(tree, source_code, file_path)

            file_name = Path(file_path).stem
            file_path_abs = str(Path(file_path).absolute())

            # 파일명과 파일 경로를 블록에 설정
            for block in blocks:
                block.file_path = file_path_abs
                if block.block_type == BlockType.MODULE:
                    block.name = file_name

            self.analyzed_blocks.extend(blocks)
            logger.debug(f"Found {len(blocks)} blocks in {file_path}")
            return blocks

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            raise

    def get_all_blocks(self) -> list[CodeBlock]:
        """분석된 모든 블록 반환

        Returns:
            지금까지 분석된 모든 CodeBlock들의 리스트
        """
        return self.analyzed_blocks

    def analyze_directory(self, dir_path: str) -> list[CodeBlock]:
        """디렉토리 내 모든 Python 파일 분석

        재귀적으로 디렉토리를 탐색하여 모든 .py 파일을 분석합니다.

        Args:
            dir_path: 분석할 디렉토리 경로

        Returns:
            추출된 모든 CodeBlock들의 리스트
        """
        logger.info(f"Analyzing directory: {dir_path}")
        python_files = self.file_reader.find_python_files(dir_path)
        all_blocks: list[CodeBlock] = []

        logger.debug(f"Found {len(python_files)} Python files")

        for py_file in python_files:
            try:
                blocks = self.analyze_file(str(py_file))
                all_blocks.extend(blocks)
            except Exception as e:
                logger.warning(f"Skipped {py_file}: {e}")

        logger.info(f"Analysis complete. Total blocks: {len(all_blocks)}")
        return all_blocks

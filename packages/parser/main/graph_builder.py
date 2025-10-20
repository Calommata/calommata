from pathlib import Path
from typing import List
from .base_parser import BaseParser
from .ast_extractor import ASTExtractor
from .code_block import CodeBlock
import tree_sitter_python as tslanguage


class CodeAnalyzer:
    """코드 분석기 - Tree-sitter를 사용하여 코드 블록 추출"""

    def __init__(self):
        self.parser = BaseParser(tslanguage.language())
        self.extractor = ASTExtractor(self.parser.language)
        self.analyzed_blocks: List[CodeBlock] = []

    def analyze_file(self, file_path: str) -> List[CodeBlock]:
        """단일 파일 분석"""
        print(f"Analyzing file: {file_path}")

        try:
            source_code = self._read_file(file_path)
            tree = self.parser.parse_code(source_code)
            blocks = self.extractor.extract_blocks(tree, source_code, file_path)

            file_name = Path(file_path).stem
            file_path_abs = str(Path(file_path).absolute())

            # 파일명과 파일 경로를 블록에 설정
            for block in blocks:
                block.file_path = file_path_abs
                if block.block_type == "module":
                    block.name = file_name

            self.analyzed_blocks.extend(blocks)
            print(f"  - Found {len(blocks)} blocks")
            return blocks

        except Exception as e:
            print(f"  - Error analyzing {file_path}: {e}")
            return []

    def _read_file(self, file_path: str) -> str:
        """파일 읽기"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def analyze_directory(self, dir_path: str) -> List[CodeBlock]:
        """디렉토리 내 모든 Python 파일 분석"""
        python_files = self._find_python_files(dir_path)
        all_blocks: List[CodeBlock] = []

        for py_file in python_files:
            try:
                blocks = self.analyze_file(str(py_file))
                all_blocks.extend(blocks)
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")

        return all_blocks

    def _find_python_files(self, dir_path: str) -> List[Path]:
        """디렉토리에서 Python 파일들 찾기"""
        path = Path(dir_path)
        return list(path.glob("**/*.py"))

    def get_all_blocks(self) -> List[CodeBlock]:
        """분석된 모든 블록 반환"""
        return self.analyzed_blocks

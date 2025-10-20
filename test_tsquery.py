"""TSQuery 기반 AST 추출기 테스트 예제"""

import tree_sitter_python
from tree_sitter import Language

from src.parser.ast_extractor import ASTExtractor
from src.parser.base_parser import BaseParser
from src.parser.queries.python_queries import PYTHON_QUERIES


def test_tsquery_extractor():
    """TSQuery 기반 추출기 테스트"""

    # Python 언어 설정
    py_language = Language(tree_sitter_python.language())

    # 테스트 코드
    test_code = '''
import os
from typing import Dict, List

class Calculator:
    """간단한 계산기 클래스"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        """두 수를 더함"""
        result = a + b
        self.history.append(f"add({a}, {b}) = {result}")
        return result
    
    def multiply(self, a: int, b: int) -> int:
        """두 수를 곱함"""
        result = a * b
        self.history.append(f"multiply({a}, {b}) = {result}")
        return result

def create_calculator() -> Calculator:
    """계산기 인스턴스 생성"""
    calc = Calculator()
    return calc

# 전역 변수
DEFAULT_CALC = create_calculator()
'''

    # 파서와 추출기 생성
    parser = BaseParser(tree_sitter_python.language())
    extractor = ASTExtractor(py_language, PYTHON_QUERIES)

    # 코드 파싱
    tree = parser.parse_code(test_code)

    # 블록 추출
    blocks = extractor.extract_blocks(tree, test_code, "test.py")

    # 결과 출력
    print("=== 추출된 코드 블록들 ===")
    for block in blocks:
        print(f"\n{block.block_type.upper()}: {block.name}")
        print(f"  라인: {block.start_line + 1}-{block.end_line + 1}")
        if block.parent:
            print(f"  부모: {block.parent.name}")
        if block.imports:
            print(f"  Import: {block.imports}")
        if block.dependencies:
            print(f"  의존성: {block.dependencies}")
        if block.docstring:
            print(f"  문서: {block.docstring}")


if __name__ == "__main__":
    test_tsquery_extractor()

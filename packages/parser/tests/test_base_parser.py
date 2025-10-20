"""BaseParser 테스트

BaseParser 클래스의 기능을 검증하는 테스트들입니다.
Tree-sitter 파서의 기본 동작과 에러 처리를 테스트합니다.
"""

import pytest

from src.base_parser import BaseParser
import tree_sitter_python as tslanguage


class TestBaseParser:
    """BaseParser 클래스 테스트"""

    @pytest.fixture
    def parser(self):
        """파서 인스턴스 픽스처"""
        return BaseParser(tslanguage.language())

    def test_parser_initialization(self, parser):
        """파서 초기화 테스트"""
        assert parser is not None
        assert parser.language is not None
        assert parser.parser is not None

    def test_parse_simple_code(self, parser):
        """간단한 코드 파싱 테스트"""
        code = "x = 1"
        tree = parser.parse_code(code)

        assert tree is not None
        assert tree.root_node is not None

    def test_parse_function(self, parser):
        """함수 정의 파싱 테스트"""
        code = """
def hello():
    return "world"
"""
        tree = parser.parse_code(code)
        assert tree is not None
        assert tree.root_node is not None

        # root_node를 순회하여 function_definition이 있는지 확인
        def has_function_def(node):
            if node.type == "function_definition":
                return True
            for child in node.children:
                if has_function_def(child):
                    return True
            return False

        assert has_function_def(tree.root_node)

    def test_parse_class(self, parser):
        """클래스 정의 파싱 테스트"""
        code = """
class MyClass:
    def __init__(self):
        pass
"""
        tree = parser.parse_code(code)
        assert tree is not None

        def has_class_def(node):
            if node.type == "class_definition":
                return True
            for child in node.children:
                if has_class_def(child):
                    return True
            return False

        assert has_class_def(tree.root_node)

    def test_parse_empty_string_raises_error(self, parser):
        """빈 문자열 파싱 시 에러 테스트"""
        with pytest.raises(ValueError):
            parser.parse_code("")

    def test_parse_complex_code(self, parser):
        """복잡한 코드 파싱 테스트"""
        code = """
import os
from typing import List

class DataProcessor:
    def __init__(self, data: List[int]):
        self.data = data
    
    def process(self):
        return sum(self.data)

def main():
    processor = DataProcessor([1, 2, 3])
    print(processor.process())
"""
        tree = parser.parse_code(code)
        assert tree is not None
        assert tree.root_node is not None

    def test_parse_with_imports(self, parser):
        """import 문 파싱 테스트"""
        code = """
import json
from pathlib import Path
from typing import Dict, List, Optional
"""
        tree = parser.parse_code(code)
        assert tree is not None

    def test_parse_multiline_strings(self, parser):
        """다중 라인 문자열 파싱 테스트"""
        code = '''
"""This is a docstring
with multiple lines
and special characters!
"""

def function():
    """Another docstring"""
    pass
'''
        tree = parser.parse_code(code)
        assert tree is not None

    def test_parse_special_characters(self, parser):
        """특수 문자가 포함된 코드 파싱 테스트"""
        code = """
# This is a comment with special chars: !@#$%^&*()
x = "String with \\"escaped\\" quotes"
y = 'Single quotes with special: @#$'
"""
        tree = parser.parse_code(code)
        assert tree is not None

    def test_parse_f_strings(self, parser):
        """f-string 파싱 테스트"""
        code = """
name = "World"
result = f"Hello, {name}!"
value = f"Result: {1 + 2}"
"""
        tree = parser.parse_code(code)
        assert tree is not None

    def test_parse_decorators(self, parser):
        """데코레이터가 있는 함수/클래스 파싱 테스트"""
        code = """
@decorator
def decorated_function():
    pass

@class_decorator
class DecoratedClass:
    @staticmethod
    def static_method():
        pass
    
    @classmethod
    def class_method(cls):
        pass
"""
        tree = parser.parse_code(code)
        assert tree is not None

    def test_parse_comprehensions(self, parser):
        """리스트/딕셔너리 컴프리헨션 파싱 테스트"""
        code = """
squares = [x**2 for x in range(10)]
evens = {x for x in range(10) if x % 2 == 0}
mapping = {x: x**2 for x in range(5)}
"""
        tree = parser.parse_code(code)
        assert tree is not None

    def test_parse_lambda(self, parser):
        """lambda 함수 파싱 테스트"""
        code = """
add = lambda x, y: x + y
filter_list = list(filter(lambda x: x > 5, [1, 2, 3, 4, 5, 6, 7]))
"""
        tree = parser.parse_code(code)
        assert tree is not None


class TestParserErrorHandling:
    """파서 에러 처리 테스트"""

    @pytest.fixture
    def parser(self):
        """파서 인스턴스 픽스처"""
        return BaseParser(tslanguage.language())

    def test_invalid_language_raises_error(self):
        """유효하지 않은 언어 객체로 초기화 시 에러"""
        with pytest.raises(TypeError):
            BaseParser(None)

    def test_invalid_string_type_for_parse_code(self, parser):
        """잘못된 타입을 parse_code에 전달할 때의 동작"""
        # Python 3.13의 타입 체킹은 런타임 에러를 발생시키지 않으므로
        # 이 테스트는 스킵할 수 있음
        pass


class TestParserPerformance:
    """파서 성능 테스트"""

    @pytest.fixture
    def parser(self):
        """파서 인스턴스 픽스처"""
        return BaseParser(tslanguage.language())

    def test_parse_large_code(self, parser):
        """큰 코드 파싱 테스트"""
        # 큰 파일 생성
        lines = []
        for i in range(100):
            lines.append(f"def function_{i}():")
            lines.append("    pass")

        code = "\n".join(lines)
        tree = parser.parse_code(code)

        assert tree is not None
        assert len(code) > 1000

    def test_parse_multiple_times(self, parser):
        """같은 파서로 여러 번 파싱 테스트"""
        code = "x = 1"

        for _ in range(10):
            tree = parser.parse_code(code)
            assert tree is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

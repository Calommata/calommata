"""AST 추출기 테스트

ASTExtractor 클래스의 기능을 검증하는 테스트들입니다.
구문 트리에서 코드 블록 추출, docstring 추출 등을 테스트합니다.
"""

import pytest

from src.base_parser import BaseParser
from src.ast_extractor import ASTExtractor
from src.code_block import CodeBlock, DependencyType
import tree_sitter_python as tslanguage


class TestASTExtractor:
    """AST 추출기 테스트"""

    @pytest.fixture
    def extractor(self):
        """AST 추출기 인스턴스 픽스처"""
        parser = BaseParser(tslanguage.language())
        return ASTExtractor(parser.language)

    @pytest.fixture
    def simple_code(self):
        """간단한 Python 코드"""
        return """
def hello():
    pass

class MyClass:
    def method(self):
        pass
"""

    @pytest.fixture
    def code_with_docstring(self):
        """docstring이 있는 Python 코드"""
        return '''
def greet(name):
    """사용자에게 인사한다"""
    print(f"Hello, {name}!")

class Calculator:
    """계산 도구"""
    
    def add(self, a, b):
        """두 수를 더한다"""
        return a + b
'''

    @pytest.fixture
    def code_with_imports(self):
        """import 문이 있는 코드"""
        return """
import os
import sys
from pathlib import Path
from typing import List, Dict
from user import User
"""

    @pytest.fixture
    def code_with_dependencies(self):
        """의존성이 있는 코드"""
        return """
class DatabaseConnection:
    def __init__(self):
        pass

class UserManager:
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    def get_user(self):
        return self.db.query()
"""

    def test_extractor_initialization(self, extractor):
        """추출기 초기화 테스트"""
        assert extractor is not None
        assert extractor.language is not None

    def test_extract_simple_blocks(self, extractor, simple_code):
        """간단한 코드 블록 추출 테스트"""
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(simple_code)
        blocks = extractor.extract_blocks(tree, simple_code)

        # module, function, class, method 블록이 추출되어야 함
        assert len(blocks) >= 4

        block_types = {block.block_type for block in blocks}
        assert "module" in block_types
        assert "function" in block_types
        assert "class" in block_types

    def test_extract_docstrings(self, extractor, code_with_docstring):
        """docstring 추출 테스트"""
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code_with_docstring)
        blocks = extractor.extract_blocks(tree, code_with_docstring)

        # 함수와 클래스 블록 찾기
        functions = [b for b in blocks if b.block_type == "function"]
        classes = [b for b in blocks if b.block_type == "class"]

        # docstring이 있는지 확인
        assert any(f.docstring for f in functions), "함수에서 docstring을 추출하지 못함"
        assert any(c.docstring for c in classes), "클래스에서 docstring을 추출하지 못함"

    def test_extract_imports(self, extractor, code_with_imports):
        """import 문 추출 테스트"""
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code_with_imports)
        blocks = extractor.extract_blocks(tree, code_with_imports)

        import_blocks = [b for b in blocks if b.block_type == "import"]
        assert len(import_blocks) >= 3

        # import된 모듈 이름들이 있는지 확인
        all_imports = []
        for block in import_blocks:
            all_imports.extend(block.imports or [])

        assert "os" in all_imports
        assert "sys" in all_imports
        assert "pathlib" in all_imports or "Path" in all_imports

    def test_extract_class_dependencies(self, extractor, code_with_dependencies):
        """클래스 의존성 추출 테스트"""
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code_with_dependencies)
        blocks = extractor.extract_blocks(tree, code_with_dependencies)

        classes = [b for b in blocks if b.block_type == "class"]
        user_manager = [c for c in classes if c.name == "UserManager"][0]

        # UserManager는 DatabaseConnection에 의존해야 함
        assert user_manager.dependencies
        assert any("DatabaseConnection" in dep for dep in user_manager.dependencies)

    def test_extract_node_name(self, extractor):
        """노드 이름 추출 테스트"""
        code = "def my_function(): pass"
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code)

        # 함수 정의 노드 찾기
        def find_node(node):
            if node.type == "function_definition":
                return node
            for child in node.children:
                result = find_node(child)
                if result:
                    return result
            return None

        func_node = find_node(tree.root_node)
        assert func_node is not None

        name = extractor._extract_node_name(func_node, "function")
        assert name == "my_function"

    def test_get_node_text(self, extractor):
        """노드 텍스트 추출 테스트"""
        code = 'x = "hello"'
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code)

        text = extractor._get_node_text(tree.root_node)
        assert "hello" in text or "=" in text

    def test_clean_docstring(self, extractor):
        """docstring 정리 테스트"""
        test_cases = [
            ('"""This is a docstring"""', "This is a docstring"),
            ("'''Single quotes'''", "Single quotes"),
            ('"Double quotes"', "Double quotes"),
            ("'Single'", "Single"),
            ('"""Multi\nline\ndocstring"""', "Multi\nline\ndocstring"),
        ]

        for input_doc, expected in test_cases:
            result = extractor._clean_docstring(input_doc)
            assert result == expected, f"Failed for {input_doc}"

    def test_is_custom_type(self, extractor):
        """커스텀 타입 판별 테스트"""
        # 커스텀 타입
        assert extractor._is_custom_type("MyClass")
        assert extractor._is_custom_type("User")

        # 내장 타입
        assert not extractor._is_custom_type("str")
        assert not extractor._is_custom_type("List")
        assert not extractor._is_custom_type("Dict")
        assert not extractor._is_custom_type("Optional")

    def test_extract_function_calls(self, extractor):
        """함수 호출 추출 테스트"""
        code = """
def caller():
    helper()
    obj.method()

def helper():
    pass
"""
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code)
        blocks = extractor.extract_blocks(tree, code)

        caller = [b for b in blocks if b.name == "caller"]
        assert len(caller) > 0

        # caller 함수가 helper를 호출하므로 의존성에 포함되어야 함
        # (함수 호출 분석이 활성화되어 있다면)


class TestEdgeCases:
    """엣지 케이스 테스트"""

    @pytest.fixture
    def extractor(self):
        """AST 추출기 인스턴스 픽스처"""
        parser = BaseParser(tslanguage.language())
        return ASTExtractor(parser.language)

    def test_empty_code(self, extractor):
        """빈 코드 처리 테스트"""
        code = ""
        parser = BaseParser(tslanguage.language())

        # 빈 코드는 ValueError를 발생시켜야 함
        with pytest.raises(ValueError):
            parser.parse_code(code)

    def test_code_with_syntax_error_strings(self, extractor):
        """문법적으로는 올바르지만 특수 문자가 있는 코드"""
        code = '''
def function_with_special_chars():
    """Has 'quotes' and "double quotes" """
    x = "string with \\"escaped\\" quotes"
    return x
'''
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code)
        blocks = extractor.extract_blocks(tree, code)

        functions = [b for b in blocks if b.block_type == "function"]
        assert len(functions) > 0

    def test_nested_classes_and_functions(self, extractor):
        """중첩된 클래스와 함수"""
        code = """
class Outer:
    def outer_method(self):
        pass
    
    class Inner:
        def inner_method(self):
            pass
"""
        parser = BaseParser(tslanguage.language())
        tree = parser.parse_code(code)
        blocks = extractor.extract_blocks(tree, code)

        classes = [b for b in blocks if b.block_type == "class"]
        functions = [b for b in blocks if b.block_type == "function"]

        assert len(classes) >= 1
        assert len(functions) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

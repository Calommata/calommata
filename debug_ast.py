"""Python AST 구조 분석 도구"""

import tree_sitter_python
from tree_sitter import Language, Parser


def print_ast_structure(node, depth=0, max_depth=5):
    """AST 구조를 출력"""
    if depth > max_depth:
        return

    indent = "  " * depth
    node_text = node.text.decode("utf-8") if node.text else ""
    # 텍스트가 너무 길면 줄임
    if len(node_text) > 50:
        node_text = node_text[:47] + "..."
    node_text = node_text.replace("\n", "\\n")

    print(f"{indent}{node.type}: '{node_text}'")

    for child in node.children:
        print_ast_structure(child, depth + 1, max_depth)


def main():
    # Python 언어 설정
    py_language = Language(tree_sitter_python.language())
    parser = Parser(py_language)

    # 간단한 테스트 코드
    test_code = '''
class Calculator:
    """간단한 계산기"""
    
    def add(self, a: int, b: int) -> int:
        """두 수를 더함"""
        return a + b

def create_calculator():
    return Calculator()
'''

    # 코드 파싱
    tree = parser.parse(bytes(test_code, "utf8"))

    print("=== Python AST 구조 ===")
    print_ast_structure(tree.root_node)


if __name__ == "__main__":
    main()

from tree_sitter import Node, Tree
from code_block import CodeBlock

PYTHON_NODE_TYPES: dict[str, str] = {
    "module": "module",
    "function_definition": "function",
    "class_definition": "class",
    "import_statement": "import",
    "import_from_statement": "import",
}


class ASTExtractor:
    """tree-sitter AST에서 코드 블록 추출"""

    def extract_blocks(self, tree: Tree) -> list[CodeBlock]:
        """AST 트리에서 모든 블록 추출"""
        blocks: list[CodeBlock] = []
        self._traverse_node(tree.root_node, None, blocks)
        return blocks

    def _traverse_node(
        self, node: Node, parent: CodeBlock | None, blocks: list[CodeBlock]
    ):
        """재귀적으로 노드 순회"""
        node_type = node.type

        if node_type in PYTHON_NODE_TYPES:
            block = self._create_block(node, parent)
            if block:
                blocks.append(block)

                # 자식 노드 처리
                for child in node.children:
                    self._traverse_node(child, block, blocks)
        else:
            # 다른 노드는 자식만 순회
            for child in node.children:
                self._traverse_node(child, parent, blocks)

    def _create_block(self, node: Node, parent: CodeBlock | None) -> CodeBlock | None:
        """노드에서 블록 생성"""
        node_type = node.type

        # 소스 코드 추출
        source_code = node.text.decode("utf-8") if node.text else ""

        if node_type == "module":
            # 모듈 블록 생성
            return CodeBlock(
                block_type="module",
                name="module",
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                parent=parent,
                source_code=source_code,
            )

        elif node_type == "function_definition":
            name = self._get_function_name(node)
            return CodeBlock(
                block_type="function",
                name=name,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                parent=parent,
                source_code=source_code,
            )

        elif node_type == "class_definition":
            name = self._get_class_name(node)
            dependencies = self._extract_dependencies(node)
            return CodeBlock(
                block_type="class",
                name=name,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                parent=parent,
                source_code=source_code,
                dependencies=dependencies,
            )

        elif node_type in ["import_statement", "import_from_statement"]:
            imports = self._extract_imports(node)
            return CodeBlock(
                block_type="import",
                name="import",
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                parent=parent,
                imports=imports,
                source_code=source_code,
            )

        return None

    def _get_function_name(self, node: Node) -> str:
        """함수명 추출"""
        # function_definition의 두 번째 자식이 함수명
        for child in node.children:
            if child.type == "identifier" and child.text is not None:
                return child.text.decode("utf-8")
        return "unknown"

    def _get_class_name(self, node: Node) -> str:
        """클래스명 추출"""
        for child in node.children:
            if child.type == "identifier" and child.text is not None:
                return child.text.decode("utf-8")
        return "unknown"

    def _extract_imports(self, node: Node) -> list[str]:
        """import 문에서 모듈명 추출"""
        imports: list[str] = []
        if node.text is None:
            return imports

        text = node.text.decode("utf-8")
        if text.startswith("from "):
            # from X import Y
            parts = text.split()
            if len(parts) >= 2:
                imports.append(parts[1])
        elif text.startswith("import "):
            # import X, Y, Z
            parts = text.replace("import ", "").split(",")
            imports.extend([p.strip().split(" as ")[0] for p in parts])

        return imports

    def _extract_dependencies(self, node: Node) -> list[str]:
        """클래스의 타입 힌트에서 의존하는 클래스명 추출"""
        dependencies: list[str] = []
        source_code = node.text.decode("utf-8") if node.text else ""

        # 타입 힌트에서 클래스명 추출 (예: AuthenticationService)
        import re

        # 함수 시그니처에서 타입 힌트 찾기
        pattern = r":\s*(\w+)\s*[,\)]"
        matches = re.findall(pattern, source_code)

        for match in matches:
            # 기본 타입과 제외
            if match not in [
                "str",
                "int",
                "float",
                "bool",
                "dict",
                "list",
                "tuple",
                "set",
                "None",
                "Any",
            ]:
                if match and match[0].isupper():  # 클래스명은 대문자로 시작
                    dependencies.append(match)

        # 인스턴스 변수 할당에서도 추출 (self.xxx = parameter)
        init_pattern = r"self\.(\w+)\s*=\s*(\w+)"
        init_matches = re.findall(init_pattern, source_code)

        for _, param_name in init_matches:
            # 함수 시그니처에서 이 파라미터의 타입 찾기
            param_pattern = rf"{param_name}\s*:\s*(\w+)"
            param_type_matches = re.findall(param_pattern, source_code)
            for ptype in param_type_matches:
                if (
                    ptype
                    and ptype[0].isupper()
                    and ptype not in ["Optional", "Union", "List", "Dict"]
                ):
                    if ptype not in dependencies:
                        dependencies.append(ptype)

        return list(set(dependencies))  # 중복 제거

from tree_sitter import Node, Tree
from typing import List
from .code_block import CodeBlock
import re


class ASTExtractor:
    """tree-sitter를 사용하여 AST에서 코드 블록 추출"""

    def __init__(self, language):
        """언어별 파서 초기화"""
        self.language = language

    def extract_blocks(
        self, tree: Tree, source_code: str, file_path: str = ""
    ) -> List[CodeBlock]:
        """AST 트리에서 모든 블록 추출"""
        blocks: List[CodeBlock] = []

        # 1. 모듈 블록 생성 (최상위)
        module_block = self._create_module_block(source_code, file_path)
        blocks.append(module_block)

        # 2. 재귀적으로 노드 순회하여 블록들 추출
        self._traverse_node(tree.root_node, module_block, blocks, file_path)

        # 3. 의존성 관계 분석
        self._analyze_function_calls(blocks, tree.root_node)

        return blocks

    def _create_module_block(self, source_code: str, file_path: str = "") -> CodeBlock:
        """모듈 블록 생성"""
        return CodeBlock(
            block_type="module",
            name="module",
            start_line=0,
            end_line=len(source_code.split("\n")) - 1,
            file_path=file_path,
            parent=None,
            source_code=source_code,
        )

    def _traverse_node(
        self,
        node: Node,
        parent: CodeBlock,
        blocks: List[CodeBlock],
        file_path: str = "",
    ):
        """재귀적으로 노드 순회하여 블록 추출"""
        # 노드 타입별 블록 생성
        block = None

        if node.type == "import_statement":
            block = self._create_import_block(node, parent, file_path)
        elif node.type == "import_from_statement":
            block = self._create_import_from_block(node, parent, file_path)
        elif node.type == "class_definition":
            block = self._create_class_block(node, parent, file_path)
        elif node.type == "function_definition":
            block = self._create_function_block(node, parent, file_path)

        # 블록이 생성되었으면 추가
        if block:
            blocks.append(block)
            # 클래스인 경우 새로운 부모로 설정
            if node.type == "class_definition":
                for child in node.children:
                    self._traverse_node(child, block, blocks, file_path)
                return

        # 자식 노드들 순회
        for child in node.children:
            self._traverse_node(child, parent, blocks, file_path)

    def _create_import_block(
        self, node: Node, parent: CodeBlock, file_path: str = ""
    ) -> CodeBlock | None:
        """import 문 블록 생성"""
        import_names = self._extract_import_names(node)
        if not import_names:
            return None

        return CodeBlock(
            block_type="import",
            name=f"import_{import_names[0]}",
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=file_path,
            parent=parent,
            source_code=self._get_node_text(node),
            imports=import_names,
        )

    def _create_import_from_block(
        self, node: Node, parent: CodeBlock, file_path: str = ""
    ) -> CodeBlock | None:
        """from ... import 문 블록 생성"""
        import_names = self._extract_from_import_names(node)
        if not import_names:
            return None

        return CodeBlock(
            block_type="import",
            name=f"import_{import_names[0]}",
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=file_path,
            parent=parent,
            source_code=self._get_node_text(node),
            imports=import_names,
        )

    def _create_class_block(
        self, node: Node, parent: CodeBlock, file_path: str = ""
    ) -> CodeBlock | None:
        """클래스 블록 생성"""
        class_name = self._extract_node_name(node, "class")
        dependencies = self._extract_class_dependencies(node)
        docstring = self._extract_docstring(node)

        return CodeBlock(
            block_type="class",
            name=class_name,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=file_path,
            parent=parent,
            source_code=self._get_node_text(node),
            dependencies=dependencies,
            docstring=docstring,
        )

    def _create_function_block(
        self, node: Node, parent: CodeBlock, file_path: str = ""
    ) -> CodeBlock | None:
        """함수 블록 생성"""
        func_name = self._extract_node_name(node, "function")
        docstring = self._extract_docstring(node)

        return CodeBlock(
            block_type="function",
            name=func_name,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            file_path=file_path,
            parent=parent,
            source_code=self._get_node_text(node),
            docstring=docstring,
        )

    def _get_node_text(self, node: Node) -> str:
        """노드에서 텍스트 추출"""
        return node.text.decode("utf-8") if node.text else ""

    def _extract_node_name(self, node: Node, node_type: str) -> str:
        """노드에서 이름 추출"""
        for child in node.children:
            if child.type == "identifier" and child.text is not None:
                return child.text.decode("utf-8")
        return f"unknown_{node_type}"

    def _extract_import_names(self, import_node: Node) -> List[str]:
        """import 문에서 모듈명들 추출"""
        source = self._get_node_text(import_node)
        if not source.startswith("import "):
            return []

        modules = source.replace("import ", "").strip()
        imports = []
        for module in modules.split(","):
            module = module.strip().split(" as ")[0].strip()  # "as alias" 부분 제거
            if module:
                imports.append(module)
        return imports

    def _extract_from_import_names(self, import_node: Node) -> List[str]:
        """from ... import 문에서 모듈명 추출"""
        source = self._get_node_text(import_node)
        if not (source.startswith("from ") and " import " in source):
            return []

        parts = source.split(" import ")
        if len(parts) >= 2:
            module_part = parts[0].replace("from ", "").strip()
            if module_part:
                return [module_part]
        return []

    def _extract_class_dependencies(self, class_node: Node) -> List[str]:
        """클래스의 의존성 추출 (상속, 타입 힌트 등)"""
        dependencies = set()
        source_code = self._get_node_text(class_node)

        # 1. 상속 관계 추출
        superclass_pattern = r"class\s+\w+\s*\(\s*([^)]+)\s*\):"
        superclass_matches = re.findall(superclass_pattern, source_code)
        for match in superclass_matches:
            superclasses = [s.strip() for s in match.split(",") if s.strip()]
            for sc in superclasses:
                if sc and sc[0].isupper():  # 클래스명은 대문자로 시작
                    dependencies.add(sc)

        # 2. 타입 힌트에서 의존성 추출
        type_hint_pattern = r":\s*([A-Z]\w*)"
        type_matches = re.findall(type_hint_pattern, source_code)
        for type_name in type_matches:
            if self._is_custom_type(type_name):
                dependencies.add(type_name)

        # 3. 인스턴스 변수 타입 힌트
        instance_var_pattern = r"self\.(\w+)\s*:\s*([A-Z]\w*)"
        instance_matches = re.findall(instance_var_pattern, source_code)
        for _, type_name in instance_matches:
            if self._is_custom_type(type_name):
                dependencies.add(type_name)

        return list(dependencies)

    def _is_custom_type(self, type_name: str) -> bool:
        """커스텀 타입인지 확인 (내장 타입 제외)"""
        builtin_types = {
            "Any",
            "Optional",
            "Union",
            "List",
            "Dict",
            "Tuple",
            "Set",
            "str",
            "int",
            "float",
            "bool",
            "bytes",
            "None",
        }
        return type_name not in builtin_types

    def _analyze_function_calls(self, blocks: List[CodeBlock], node: Node):
        """함수 호출 관계 분석"""
        if node.type == "call":
            self._process_function_call(node, blocks)

        # 자식 노드들 재귀 순회
        for child in node.children:
            self._analyze_function_calls(blocks, child)

    def _process_function_call(self, call_node: Node, blocks: List[CodeBlock]):
        """개별 함수 호출 처리"""
        call_line = call_node.start_point[0]
        caller_block = self._find_containing_block(call_line, blocks)

        if not caller_block:
            return

        called_func = self._extract_called_function_name(call_node)
        if called_func and called_func not in caller_block.dependencies:
            caller_block.dependencies.append(called_func)

    def _find_containing_block(
        self, line: int, blocks: List[CodeBlock]
    ) -> CodeBlock | None:
        """특정 라인을 포함하는 가장 구체적인 블록 찾기"""
        candidates = [b for b in blocks if b.start_line <= line <= b.end_line]
        if not candidates:
            return None
        # 가장 작은 범위의 블록 반환 (함수 > 클래스 > 모듈)
        return min(candidates, key=lambda b: b.end_line - b.start_line)

    def _extract_called_function_name(self, call_node: Node) -> str | None:
        """함수 호출 노드에서 호출되는 함수명 추출"""
        source = self._get_node_text(call_node)
        if not source or "(" not in source:
            return None

        func_part = source.split("(")[0].strip()
        if "." in func_part:
            # 메서드 호출: obj.method()
            return func_part.split(".")[-1]
        else:
            # 함수 호출: func()
            return func_part

    def _extract_docstring(self, node: Node) -> str | None:
        """클래스나 함수에서 docstring 추출"""
        # 클래스나 함수 정의 바로 다음에 오는 string literal을 찾기
        for child in node.children:
            if child.type == "block":  # 함수/클래스 본문
                for grandchild in child.children:
                    if grandchild.type == "expression_statement":
                        # expression_statement 안에서 string을 찾기
                        for great_grandchild in grandchild.children:
                            if great_grandchild.type == "string":
                                docstring_text = self._get_node_text(great_grandchild)
                                # 따옴표 제거 및 정리
                                return self._clean_docstring(docstring_text)
                        break  # 첫 번째 statement만 확인
                break
        return None

    def _clean_docstring(self, docstring_text: str) -> str:
        """docstring 텍스트 정리"""
        if not docstring_text:
            return ""

        # 따옴표 제거 (""", ''', ", ')
        cleaned = docstring_text.strip()
        if cleaned.startswith('"""') and cleaned.endswith('"""'):
            cleaned = cleaned[3:-3]
        elif cleaned.startswith("'''") and cleaned.endswith("'''"):
            cleaned = cleaned[3:-3]
        elif cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        # 앞뒤 공백 제거 및 개행 정리
        return cleaned.strip()

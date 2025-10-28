"""AST 추출기 - Tree-sitter TSQuery를 사용하여 AST에서 코드 블록 추출"""

import logging

from tree_sitter import Language, Node, Query, QueryCursor, Tree

from .code_block import CodeBlock, BlockType

logger = logging.getLogger(__name__)


class ASTExtractor:
    """tree-sitter TSQuery를 사용하여 AST에서 코드 블록 추출

    Tree-sitter의 구문 트리를 TSQuery로 분석하여 함수, 클래스, import 문 등의
    코드 블록을 추출합니다. 언어별 쿼리를 통해 다양한 언어를 지원합니다.
    """

    def __init__(self, language: Language, queries: dict[str, str]):
        """초기화

        Args:
            language: Tree-sitter 언어 객체
            queries: 언어별 TSQuery 쿼리 딕셔너리
        """
        self.language = language
        self.queries = self._compile_queries(queries)
        logger.debug(f"ASTExtractor initialized with language: {language}")

    def _compile_queries(self, queries: dict[str, str]) -> dict[str, Query]:
        """TSQuery 문자열을 Query 객체로 컴파일"""
        compiled = {}
        for name, query_str in queries.items():
            try:
                compiled[name] = Query(self.language, query_str)
                logger.debug(f"Compiled query '{name}' successfully")
            except Exception as e:
                logger.error(f"Failed to compile query '{name}': {e}")
                raise
        return compiled

    def extract_blocks(
        self, tree: Tree, source_code: str, file_path: str = ""
    ) -> list[CodeBlock]:
        """TSQuery를 사용하여 AST 트리에서 모든 블록 추출"""
        blocks: list[CodeBlock] = []

        # 1. 모듈 블록 생성 (최상위)
        module_block = self._create_module_block(source_code, file_path)
        blocks.append(module_block)

        # 2. TSQuery를 사용하여 각 블록 타입별 추출 (의존성 포함)
        self._extract_blocks_by_query(tree, blocks, file_path)

        return blocks

    def _extract_blocks_by_query(
        self,
        tree: Tree,
        blocks: list[CodeBlock],
        file_path: str,
    ) -> None:
        """TSQuery를 사용하여 블록들을 추출"""
        extraction_order = ["classes", "functions", "imports"]

        for query_name in extraction_order:
            if query_name not in self.queries:
                continue

            query = self.queries[query_name]
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)

            for pattern_index, captures in matches:
                # 각 쿼리별 블록 생성
                block = self._create_block_from_captures(
                    query_name, captures, blocks, file_path
                )
                if block:
                    blocks.append(block)

    def _create_block_from_captures(
        self,
        query_name: str,
        captures: dict[str, list[Node]],
        blocks: list[CodeBlock],
        file_path: str,
    ) -> CodeBlock | None:
        """TSQuery 캡처 결과로부터 CodeBlock 생성"""
        if not captures:
            return None

        # 블록 타입별 처리
        if query_name == "functions":
            return self._create_function_block_from_captures(
                captures, blocks, file_path
            )
        elif query_name == "classes":
            return self._create_class_block_from_captures(captures, blocks, file_path)
        elif query_name == "imports":
            return self._create_import_block_from_captures(captures, blocks, file_path)

        return None

    def _create_module_block(self, source_code: str, file_path: str = "") -> CodeBlock:
        """모듈 블록 생성"""
        return CodeBlock(
            block_type=BlockType.MODULE,
            name="module",
            file_path=file_path,
            parent=None,
            source_code=source_code,
        )

    def _find_parent_block(
        self, blocks: list[CodeBlock], current_name: str = ""
    ) -> CodeBlock | None:
        """부모 블록 찾기

        단순화된 로직:
        - 마지막에 추가된 CLASS 블록을 찾아서 부모로 사용 (method인 경우)
        - 없으면 MODULE을 부모로 사용
        """
        # 역순으로 검색하여 가장 최근의 클래스 블록 찾기
        for block in reversed(blocks):
            if block.block_type == BlockType.CLASS:
                return block

        # 클래스가 없으면 모듈 블록 반환
        for block in blocks:
            if block.block_type == BlockType.MODULE:
                return block

        return None

    def _get_node_text(self, node: Node) -> str:
        """노드에서 텍스트 추출"""
        return node.text.decode("utf-8") if node.text else ""

    def _create_import_block_from_captures(
        self,
        captures: dict[str, list[Node]],
        blocks: list[CodeBlock],
        file_path: str,
    ) -> CodeBlock | None:
        """TSQuery 캡처로부터 import 블록 생성"""
        # 모듈명 추출
        module_names = []
        main_node = None
        import_node = None

        if "import.module" in captures and captures["import.module"]:
            main_node = captures["import.module"][0]
            module_name = self._get_node_text(main_node)
            module_names.append(module_name)

            # import 문 전체 노드 찾기
            import_node = main_node.parent
            while import_node and import_node.type not in [
                "import_statement",
                "import_from_statement",
            ]:
                import_node = import_node.parent

        if not main_node or not import_node:
            return None

        source_code = self._get_node_text(import_node)
        parent = self._find_parent_block(blocks)

        block_name = f"import_{module_names[0]}" if module_names else "import_unknown"

        return CodeBlock(
            block_type=BlockType.IMPORT,
            name=block_name,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            imports=module_names,
        )

    def _create_class_block_from_captures(
        self,
        captures: dict[str, list[Node]],
        blocks: list[CodeBlock],
        file_path: str,
    ) -> CodeBlock | None:
        """TSQuery 캡처로부터 클래스 블록 생성"""
        if "class.name" not in captures or not captures["class.name"]:
            return None

        name_node = captures["class.name"][0]
        class_name = self._get_node_text(name_node)

        # 클래스 정의 전체 노드 찾기 (name_node의 부모들 중에서)
        class_def_node = name_node.parent
        while class_def_node and class_def_node.type != "class_definition":
            class_def_node = class_def_node.parent

        if not class_def_node:
            return None

        source_code = self._get_node_text(class_def_node)

        parent = self._find_parent_block(blocks)

        # 상속 정보 추출
        dependencies = []
        if "class.superclass" in captures:
            for superclass_node in captures["class.superclass"]:
                superclass = self._get_node_text(superclass_node)
                if superclass:
                    dependencies.append(f"inherits:{superclass}")

        # 클래스 내부의 의존성 추출
        class_dependencies = self._extract_dependencies_from_node(class_def_node)
        dependencies.extend(class_dependencies)

        return CodeBlock(
            block_type=BlockType.CLASS,
            name=class_name,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            dependencies=dependencies,
        )

    def _create_function_block_from_captures(
        self,
        captures: dict[str, list[Node]],
        blocks: list[CodeBlock],
        file_path: str,
    ) -> CodeBlock | None:
        """TSQuery 캡처로부터 함수 블록 생성"""
        if "function.name" not in captures or not captures["function.name"]:
            return None

        name_node = captures["function.name"][0]
        func_name = self._get_node_text(name_node)

        # 함수 정의 전체 노드 찾기 (name_node의 부모들 중에서)
        func_def_node = name_node.parent
        while func_def_node and func_def_node.type != "function_definition":
            func_def_node = func_def_node.parent

        if not func_def_node:
            return None

        source_code = self._get_node_text(func_def_node)

        parent = self._find_parent_block(blocks)

        # 함수 내부의 의존성 추출 (함수 호출, 변수 사용 등)
        dependencies = self._extract_dependencies_from_node(func_def_node)

        return CodeBlock(
            block_type=BlockType.FUNCTION,
            name=func_name,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            dependencies=dependencies,
        )

    def _extract_dependencies_from_node(self, node: Node) -> list[str]:
        """노드 내부에서 의존성 추출 (TSQuery 활용)

        Args:
            node: 분석할 AST 노드 (function_definition, class_definition 등)

        Returns:
            의존성 문자열 리스트
        """
        dependencies = []

        # 1. 함수 호출 추출
        if "function_calls" in self.queries:
            query = self.queries["function_calls"]
            cursor = QueryCursor(query)
            matches = cursor.matches(node)

            for pattern_index, captures in matches:
                if "call.function" in captures:
                    for call_node in captures["call.function"]:
                        func_name = self._get_node_text(call_node)
                        if func_name and f"calls:{func_name}" not in dependencies:
                            dependencies.append(f"calls:{func_name}")

                if "call.method" in captures:
                    for method_node in captures["call.method"]:
                        method_name = self._get_node_text(method_node)
                        if method_name and f"calls:{method_name}" not in dependencies:
                            dependencies.append(f"calls:{method_name}")

        # 2. 변수 정의 추출
        if "variable_usage" in self.queries:
            query = self.queries["variable_usage"]
            cursor = QueryCursor(query)
            matches = cursor.matches(node)

            for pattern_index, captures in matches:
                if "variable.name" in captures:
                    for var_node in captures["variable.name"]:
                        var_name = self._get_node_text(var_node)
                        if var_name and f"defines:{var_name}" not in dependencies:
                            dependencies.append(f"defines:{var_name}")

        # 3. 타입 힌트 추출
        if "type_hints" in self.queries:
            query = self.queries["type_hints"]
            cursor = QueryCursor(query)
            matches = cursor.matches(node)

            for pattern_index, captures in matches:
                if "type.name" in captures:
                    for type_node in captures["type.name"]:
                        type_name = self._get_node_text(type_node)
                        if type_name and f"type:{type_name}" not in dependencies:
                            dependencies.append(f"type:{type_name}")

        return dependencies

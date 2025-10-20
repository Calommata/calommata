"""AST 추출기 - Tree-sitter TSQuery를 사용하여 AST에서 코드 블록 추출"""

import logging

from tree_sitter import Language, Node, Query, QueryCursor, Tree

from .code_block import CodeBlock

logger = logging.getLogger(__name__)


class ASTExtractor:
    """tree-sitter TSQuery를 사용하여 AST에서 코드 블록 추출

    Tree-sitter의 구문 트리를 TSQuery로 분석하여 함수, 클래스, import 문 등의
    코드 블록을 추출합니다. 언어별 쿼리를 통해 다양한 언어를 지원합니다.
    """

    def __init__(self, language: Language, queries: dict[str, str]) -> None:
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
        source_lines = source_code.split("\n")

        # 1. 모듈 블록 생성 (최상위)
        module_block = self._create_module_block(source_code, file_path)
        blocks.append(module_block)

        # 2. TSQuery를 사용하여 각 블록 타입별 추출
        self._extract_blocks_by_query(tree, blocks, source_lines, file_path)

        # 3. 의존성 관계 분석
        self._analyze_dependencies(tree, blocks, source_lines)

        return blocks

    def _extract_blocks_by_query(
        self,
        tree: Tree,
        blocks: list[CodeBlock],
        source_lines: list[str],
        file_path: str,
    ) -> None:
        """TSQuery를 사용하여 블록들을 추출"""
        # 순서대로 처리: 클래스 먼저, 그 다음 함수들
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
                    query_name, captures, blocks, source_lines, file_path
                )
                if block:
                    blocks.append(block)

    def _create_block_from_captures(
        self,
        query_name: str,
        captures: dict[str, list[Node]],
        blocks: list[CodeBlock],
        source_lines: list[str],
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
            block_type="module",
            name="module",
            start_line=0,
            end_line=len(source_code.split("\n")) - 1,
            file_path=file_path,
            parent=None,
            source_code=source_code,
        )

    def _find_parent_block(
        self, line: int, blocks: list[CodeBlock]
    ) -> CodeBlock | None:
        """특정 라인을 포함하는 가장 구체적인 부모 블록 찾기"""
        candidates = [
            b
            for b in blocks
            if b.start_line <= line <= b.end_line and b.block_type != "import"
        ]
        if not candidates:
            return None
        # 가장 작은 범위의 블록 반환 (함수 > 클래스 > 모듈)
        return min(candidates, key=lambda b: b.end_line - b.start_line)

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

        start_line = import_node.start_point[0]
        end_line = import_node.end_point[0]
        source_code = self._get_node_text(import_node)
        parent = self._find_parent_block(start_line, blocks)

        block_name = f"import_{module_names[0]}" if module_names else "import_unknown"

        return CodeBlock(
            block_type="import",
            name=block_name,
            start_line=start_line,
            end_line=end_line,
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

        start_line = class_def_node.start_point[0]
        end_line = class_def_node.end_point[0]
        source_code = self._get_node_text(class_def_node)

        parent = self._find_parent_block(start_line, blocks)

        # 상속 정보 추출
        dependencies = []
        if "class.superclass" in captures:
            for superclass_node in captures["class.superclass"]:
                superclass = self._get_node_text(superclass_node)
                if superclass:
                    dependencies.append(f"inherits:{superclass}")

        # docstring 추출
        docstring = None
        if "class.docstring" in captures and captures["class.docstring"]:
            docstring_text = self._get_node_text(captures["class.docstring"][0])
            docstring = self._clean_docstring(docstring_text)

        return CodeBlock(
            block_type="class",
            name=class_name,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            dependencies=dependencies,
            docstring=docstring,
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

        start_line = func_def_node.start_point[0]
        end_line = func_def_node.end_point[0]
        source_code = self._get_node_text(func_def_node)

        parent = self._find_parent_block(start_line, blocks)

        # docstring 추출
        docstring = None
        if "function.docstring" in captures and captures["function.docstring"]:
            docstring_text = self._get_node_text(captures["function.docstring"][0])
            docstring = self._clean_docstring(docstring_text)

        return CodeBlock(
            block_type="function",
            name=func_name,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            parent=parent,
            source_code=source_code,
            docstring=docstring,
        )

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

    def _analyze_dependencies(
        self, tree: Tree, blocks: list[CodeBlock], source_lines: list[str]
    ) -> None:
        """TSQuery를 사용하여 의존성 관계 분석"""
        # 함수 호출 관계 분석
        if "function_calls" in self.queries:
            self._analyze_function_calls_with_query(tree, blocks)

        # 변수 사용 관계 분석
        if "variable_usage" in self.queries:
            self._analyze_variable_usage_with_query(tree, blocks)

        # 타입 힌트 관계 분석
        if "type_hints" in self.queries:
            self._analyze_type_hints_with_query(tree, blocks)

    def _analyze_function_calls_with_query(
        self, tree: Tree, blocks: list[CodeBlock]
    ) -> None:
        """TSQuery를 사용하여 함수 호출 관계 분석"""
        query = self.queries["function_calls"]
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        for pattern_index, captures in matches:
            if "call.function" in captures and captures["call.function"]:
                call_node = captures["call.function"][0]
                call_line = call_node.start_point[0]

                # 호출하는 블록 찾기
                caller_block = self._find_containing_block(call_line, blocks)
                if not caller_block:
                    continue

                # 호출되는 함수명 추출
                called_func = self._get_node_text(call_node)

                if called_func and caller_block.dependencies is not None:
                    if called_func not in caller_block.dependencies:
                        caller_block.dependencies.append(called_func)

            # 메서드 호출도 처리
            if "call.method" in captures and captures["call.method"]:
                method_node = captures["call.method"][0]
                method_line = method_node.start_point[0]

                caller_block = self._find_containing_block(method_line, blocks)
                if not caller_block:
                    continue

                method_name = self._get_node_text(method_node)

                if method_name and caller_block.dependencies is not None:
                    if method_name not in caller_block.dependencies:
                        caller_block.dependencies.append(method_name)

    def _analyze_variable_usage_with_query(
        self, tree: Tree, blocks: list[CodeBlock]
    ) -> None:
        """TSQuery를 사용하여 변수 사용 관계 분석"""
        query = self.queries["variable_usage"]
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        for pattern_index, captures in matches:
            if "variable.name" in captures and captures["variable.name"]:
                var_node = captures["variable.name"][0]
                var_line = var_node.start_point[0]

                # 변수를 정의하는 블록 찾기
                defining_block = self._find_containing_block(var_line, blocks)
                if not defining_block:
                    continue

                # 변수명 추출
                var_name = self._get_node_text(var_node)
                if var_name and defining_block.dependencies is not None:
                    dep_name = f"defines:{var_name}"
                    if dep_name not in defining_block.dependencies:
                        defining_block.dependencies.append(dep_name)

    def _analyze_type_hints_with_query(
        self, tree: Tree, blocks: list[CodeBlock]
    ) -> None:
        """TSQuery를 사용하여 타입 힌트 관계 분석"""
        query = self.queries["type_hints"]
        cursor = QueryCursor(query)
        matches = cursor.matches(tree.root_node)

        for pattern_index, captures in matches:
            if "type.name" in captures and captures["type.name"]:
                type_node = captures["type.name"][0]
                type_line = type_node.start_point[0]

                # 타입 힌트를 사용하는 블록 찾기
                using_block = self._find_containing_block(type_line, blocks)
                if not using_block:
                    continue

                # 타입명 추출
                type_name = self._get_node_text(type_node)
                if type_name and using_block.dependencies is not None:
                    dep_name = f"type:{type_name}"
                    if dep_name not in using_block.dependencies:
                        using_block.dependencies.append(dep_name)

    def _find_containing_block(
        self, line: int, blocks: list[CodeBlock]
    ) -> CodeBlock | None:
        """특정 라인을 포함하는 가장 구체적인 블록 찾기"""
        candidates = [b for b in blocks if b.start_line <= line <= b.end_line]
        if not candidates:
            return None
        # 가장 작은 범위의 블록 반환 (함수 > 클래스 > 모듈)
        return min(candidates, key=lambda b: b.end_line - b.start_line)

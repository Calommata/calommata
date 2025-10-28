"""AST 추출기 - Tree-sitter TSQuery를 사용하여 AST에서 코드 블록 추출"""

import logging

from tree_sitter import Language, Node, Query, QueryCursor, Tree

from .code_block import CodeBlock
from .block_factory import BlockFactory, BlockParentResolver
from .dependency_extractor import DependencyExtractor

logger = logging.getLogger(__name__)


class QueryCompiler:
    """TSQuery 컴파일 책임만 담당"""

    @staticmethod
    def compile_queries(
        language: Language, queries: dict[str, str]
    ) -> dict[str, Query]:
        """TSQuery 문자열을 Query 객체로 컴파일

        Args:
            language: Tree-sitter 언어 객체
            queries: 언어별 TSQuery 쿼리 딕셔너리

        Returns:
            컴파일된 Query 객체 딕셔너리

        Raises:
            Exception: 쿼리 컴파일 실패 시
        """
        compiled = {}
        for name, query_str in queries.items():
            try:
                compiled[name] = Query(language, query_str)
                logger.debug(f"Compiled query '{name}' successfully")
            except Exception as e:
                logger.error(f"Failed to compile query '{name}': {e}")
                raise
        return compiled


class CaptureHandler:
    """TSQuery 캡처 처리 책임만 담당"""

    def __init__(self, dependency_extractor: DependencyExtractor):
        self.dependency_extractor = dependency_extractor

    def handle_import_capture(
        self, captures: dict[str, list[Node]], blocks: list[CodeBlock], file_path: str
    ) -> CodeBlock | None:
        """import 캡처 처리"""
        if "import.module" not in captures or not captures["import.module"]:
            return None

        main_node = captures["import.module"][0]
        module_name = self._get_node_text(main_node)
        import_node = main_node.parent

        while import_node and import_node.type not in [
            "import_statement",
            "import_from_statement",
        ]:
            import_node = import_node.parent

        if not import_node:
            return None

        source_code = self._get_node_text(import_node)
        parent = BlockParentResolver.find_parent_block(blocks)

        return BlockFactory.create_import_block(
            module_names=[module_name],
            source_code=source_code,
            file_path=file_path,
            parent=parent,
        )

    def handle_class_capture(
        self, captures: dict[str, list[Node]], blocks: list[CodeBlock], file_path: str
    ) -> CodeBlock | None:
        """클래스 캡처 처리"""
        if "class.name" not in captures or not captures["class.name"]:
            return None

        name_node = captures["class.name"][0]
        class_name = self._get_node_text(name_node)
        class_def_node = name_node.parent

        while class_def_node and class_def_node.type != "class_definition":
            class_def_node = class_def_node.parent

        if not class_def_node:
            return None

        source_code = self._get_node_text(class_def_node)
        parent = BlockParentResolver.find_parent_block(blocks)

        dependencies = []
        if "class.superclass" in captures:
            for superclass_node in captures["class.superclass"]:
                superclass = self._get_node_text(superclass_node)
                if superclass:
                    dependencies.append(f"inherits:{superclass}")

        class_dependencies = self.dependency_extractor.extract_dependencies(
            class_def_node
        )
        dependencies.extend(class_dependencies)

        return BlockFactory.create_class_block(
            class_name=class_name,
            source_code=source_code,
            file_path=file_path,
            parent=parent,
            dependencies=dependencies,
        )

    def handle_function_capture(
        self, captures: dict[str, list[Node]], blocks: list[CodeBlock], file_path: str
    ) -> CodeBlock | None:
        """함수 캡처 처리"""
        if "function.name" not in captures or not captures["function.name"]:
            return None

        name_node = captures["function.name"][0]
        func_name = self._get_node_text(name_node)
        func_def_node = name_node.parent

        while func_def_node and func_def_node.type != "function_definition":
            func_def_node = func_def_node.parent

        if not func_def_node:
            return None

        source_code = self._get_node_text(func_def_node)
        parent = BlockParentResolver.find_parent_block(blocks)
        dependencies = self.dependency_extractor.extract_dependencies(func_def_node)

        return BlockFactory.create_function_block(
            func_name=func_name,
            source_code=source_code,
            file_path=file_path,
            parent=parent,
            dependencies=dependencies,
        )

    @staticmethod
    def _get_node_text(node: Node) -> str:
        """노드에서 텍스트 추출"""
        return node.text.decode("utf-8") if node.text else ""


class ASTExtractor:
    """tree-sitter TSQuery를 사용하여 AST에서 코드 블록 추출

    Tree-sitter의 구문 트리를 TSQuery로 분석하여 함수, 클래스, import 문 등의
    코드 블록을 추출합니다. 언어별 쿼리를 통해 다양한 언어를 지원합니다.

    책임:
    - TSQuery를 사용한 AST 블록 추출 조율
    """

    def __init__(self, language: Language, queries: dict[str, str]):
        """초기화

        Args:
            language: Tree-sitter 언어 객체
            queries: 언어별 TSQuery 쿼리 딕셔너리
        """
        self.language = language
        self.compiled_queries = QueryCompiler.compile_queries(language, queries)
        self.dependency_extractor = DependencyExtractor(self.compiled_queries)
        self.capture_handler = CaptureHandler(self.dependency_extractor)
        logger.debug(f"ASTExtractor initialized with language: {language}")

    def extract_blocks(
        self, tree: Tree, source_code: str, file_path: str = ""
    ) -> list[CodeBlock]:
        """TSQuery를 사용하여 AST 트리에서 모든 블록 추출

        Args:
            tree: 파싱된 구문 트리
            source_code: 원본 소스 코드
            file_path: 파일 경로

        Returns:
            추출된 CodeBlock 리스트
        """
        blocks: list[CodeBlock] = []

        module_block = BlockFactory.create_module_block(source_code, file_path)
        blocks.append(module_block)

        self._extract_blocks_by_query(tree, blocks, file_path)

        return blocks

    def _extract_blocks_by_query(
        self, tree: Tree, blocks: list[CodeBlock], file_path: str
    ) -> None:
        """TSQuery를 사용하여 블록들을 추출

        Args:
            tree: 파싱된 구문 트리
            blocks: 현재까지의 블록 리스트
            file_path: 파일 경로
        """
        extraction_order = ["classes", "functions", "imports"]

        for query_name in extraction_order:
            if query_name not in self.compiled_queries:
                continue

            query = self.compiled_queries[query_name]
            cursor = QueryCursor(query)
            matches = cursor.matches(tree.root_node)

            for pattern_index, captures in matches:
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
        """TSQuery 캡처 결과로부터 CodeBlock 생성

        Args:
            query_name: 쿼리 이름
            captures: 캡처된 노드들
            blocks: 현재까지의 블록 리스트
            file_path: 파일 경로

        Returns:
            생성된 CodeBlock 또는 None
        """
        if not captures:
            return None

        if query_name == "functions":
            return self.capture_handler.handle_function_capture(
                captures, blocks, file_path
            )
        elif query_name == "classes":
            return self.capture_handler.handle_class_capture(
                captures, blocks, file_path
            )
        elif query_name == "imports":
            return self.capture_handler.handle_import_capture(
                captures, blocks, file_path
            )

        return None

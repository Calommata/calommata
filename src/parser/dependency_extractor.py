"""의존성 추출 전담 모듈"""

import logging
from tree_sitter import Node, Query, QueryCursor

logger = logging.getLogger(__name__)


class DependencyExtractor:
    """노드 내부에서 의존성 추출 책임만 담당

    TSQuery를 활용하여 함수 호출, 변수 사용, 타입 힌트 등의 의존성을 추출합니다.
    """

    def __init__(self, queries: dict[str, Query]):
        """초기화

        Args:
            queries: 컴파일된 TSQuery 쿼리 딕셔너리
        """
        self.queries = queries
        logger.debug("DependencyExtractor initialized")

    def extract_dependencies(self, node: Node) -> list[str]:
        """노드 내부에서 의존성 추출 (TSQuery 활용)

        Args:
            node: 분석할 AST 노드 (function_definition, class_definition 등)

        Returns:
            의존성 문자열 리스트
        """
        dependencies = []

        # 1. 함수 호출 추출
        dependencies.extend(self._extract_function_calls(node))

        # 2. 변수 정의 추출
        dependencies.extend(self._extract_variable_usage(node))

        # 3. 타입 힌트 추출
        dependencies.extend(self._extract_type_hints(node))

        return dependencies

    def _extract_function_calls(self, node: Node) -> list[str]:
        """함수 호출 의존성 추출

        Args:
            node: 분석할 AST 노드

        Returns:
            함수 호출 의존성 리스트
        """
        dependencies = []

        if "function_calls" not in self.queries:
            return dependencies

        try:
            query = self.queries["function_calls"]
            cursor = QueryCursor(query)
            matches = cursor.matches(node)

            for pattern_index, captures in matches:
                if "call.function" in captures:
                    for call_node in captures["call.function"]:
                        func_name = self._get_node_text(call_node)
                        dep_str = f"calls:{func_name}"
                        if func_name and dep_str not in dependencies:
                            dependencies.append(dep_str)

                if "call.method" in captures:
                    for method_node in captures["call.method"]:
                        method_name = self._get_node_text(method_node)
                        dep_str = f"calls:{method_name}"
                        if method_name and dep_str not in dependencies:
                            dependencies.append(dep_str)

        except Exception as e:
            logger.debug(f"Error extracting function calls: {e}")

        return dependencies

    def _extract_variable_usage(self, node: Node) -> list[str]:
        """변수 사용 의존성 추출

        Args:
            node: 분석할 AST 노드

        Returns:
            변수 사용 의존성 리스트
        """
        dependencies = []

        if "variable_usage" not in self.queries:
            return dependencies

        try:
            query = self.queries["variable_usage"]
            cursor = QueryCursor(query)
            matches = cursor.matches(node)

            for pattern_index, captures in matches:
                if "variable.name" in captures:
                    for var_node in captures["variable.name"]:
                        var_name = self._get_node_text(var_node)
                        dep_str = f"defines:{var_name}"
                        if var_name and dep_str not in dependencies:
                            dependencies.append(dep_str)

        except Exception as e:
            logger.debug(f"Error extracting variable usage: {e}")

        return dependencies

    def _extract_type_hints(self, node: Node) -> list[str]:
        """타입 힌트 의존성 추출

        Args:
            node: 분석할 AST 노드

        Returns:
            타입 힌트 의존성 리스트
        """
        dependencies = []

        if "type_hints" not in self.queries:
            return dependencies

        try:
            query = self.queries["type_hints"]
            cursor = QueryCursor(query)
            matches = cursor.matches(node)

            for pattern_index, captures in matches:
                if "type.name" in captures:
                    for type_node in captures["type.name"]:
                        type_name = self._get_node_text(type_node)
                        dep_str = f"type:{type_name}"
                        if type_name and dep_str not in dependencies:
                            dependencies.append(dep_str)

        except Exception as e:
            logger.debug(f"Error extracting type hints: {e}")

        return dependencies

    @staticmethod
    def _get_node_text(node: Node) -> str:
        """노드에서 텍스트 추출

        Args:
            node: Tree-sitter 노드

        Returns:
            노드의 텍스트 내용
        """
        return node.text.decode("utf-8") if node.text else ""

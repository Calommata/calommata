"""구조적 관계 생성 전문 클래스

CONTAINS 관계 (클래스->메서드, 모듈->최상위요소)를 생성하는 책임만 담당합니다.
"""

import logging
from typing import Any

from src.graph.db import CodeGraph, CodeRelation, RelationType

logger = logging.getLogger(__name__)


class StructuralRelationBuilder:
    """구조적 관계 생성 전문 클래스"""

    def build(
        self,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
    ) -> None:
        """구조적 관계 생성 (CONTAINS)

        클래스 -> 메서드, 모듈 -> 최상위 요소 관계를 생성합니다.

        Args:
            blocks: CodeBlock 리스트
            graph: 대상 CodeGraph
            node_map: 노드 매핑
        """
        for block in blocks:
            file_path = getattr(block, "file_path", "unknown.py")
            block_type = self._get_block_type_value(block)

            if block_type == "class":
                self._create_class_contains_relations(
                    block, blocks, graph, node_map, file_path
                )
            elif block_type == "module":
                self._create_module_contains_relations(
                    block, blocks, graph, node_map, file_path
                )

    def _create_class_contains_relations(
        self,
        block: Any,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
        file_path: str,
    ) -> None:
        """클래스 -> 메서드 CONTAINS 관계 생성"""
        for other_block in blocks:
            other_type = self._get_block_type_value(other_block)

            if (
                other_type == "function"
                and getattr(other_block, "file_path", "") == file_path
                and getattr(other_block, "parent") == block
            ):
                source_key = f"{file_path}:{block.name}"
                target_key = f"{file_path}:{other_block.name}"

                if source_key in node_map and target_key in node_map:
                    relation = CodeRelation(
                        from_node_id=node_map[source_key],
                        to_node_id=node_map[target_key],
                        relation_type=RelationType.CONTAINS,
                        context="structural: class contains method",
                    )
                    self._add_relation_safe(graph, relation)

    def _create_module_contains_relations(
        self,
        block: Any,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
        file_path: str,
    ) -> None:
        """모듈 -> 최상위 요소 CONTAINS 관계 생성"""
        for other_block in blocks:
            other_type = self._get_block_type_value(other_block)

            if (
                getattr(other_block, "file_path", "") == file_path
                and other_type in ["class", "function"]
                and other_block != block
            ):
                # 최상위 요소인지 확인
                if self._is_top_level_element(other_block, blocks, file_path):
                    source_key = f"{file_path}:{block.name}"
                    target_key = f"{file_path}:{other_block.name}"

                    if source_key in node_map and target_key in node_map:
                        relation = CodeRelation(
                            from_node_id=node_map[source_key],
                            to_node_id=node_map[target_key],
                            relation_type=RelationType.CONTAINS,
                            context="structural: module contains top-level element",
                        )
                        self._add_relation_safe(graph, relation)

    @staticmethod
    def _is_top_level_element(
        element: Any,
        blocks: list[Any],
        file_path: str,
    ) -> bool:
        """요소가 최상위인지 확인 (다른 클래스 안에 없음)"""
        for container_block in blocks:
            container_type = StructuralRelationBuilder._get_block_type_value(
                container_block
            )

            if (
                container_type == "class"
                and container_block != element
                and getattr(container_block, "file_path", "") == file_path
                and getattr(element, "parent") == container_block
            ):
                return False
        return True

    @staticmethod
    def _get_block_type_value(block: Any) -> str:
        """블록의 타입 값을 문자열로 가져오기"""
        block_type = block.block_type
        return block_type.value if hasattr(block_type, "value") else str(block_type)

    @staticmethod
    def _add_relation_safe(graph: CodeGraph, relation: CodeRelation) -> None:
        """안전하게 관계 추가 (예외 무시)"""
        try:
            graph.add_relation(relation)
        except ValueError:
            pass

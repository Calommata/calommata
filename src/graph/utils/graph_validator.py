import logging
from typing import Any

from src.graph.db import CodeGraph, NodeType, RelationType

logger = logging.getLogger(__name__)


class GraphValidator:
    """그래프 유효성 검증 클래스

    그래프의 노드, 관계, 일관성을 검증하고
    오류 및 경고를 리포트합니다.
    """

    def __init__(self, graph: CodeGraph) -> None:
        """초기화

        Args:
            graph: 검증할 CodeGraph
        """
        self.graph = graph
        self.errors: list[str] = []
        self.warnings: list[str] = []
        logger.debug(f"GraphValidator initialized for graph: {graph.project_name}")

    def validate(self) -> dict[str, Any]:
        """전체 그래프 유효성 검증

        Returns:
            검증 결과 딕셔너리
        """
        self.errors.clear()
        self.warnings.clear()

        self._validate_nodes()
        self._validate_relations()
        self._validate_consistency()

        logger.info(
            f"Validation complete: {len(self.errors)} errors, {len(self.warnings)} warnings"
        )

        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self._get_validation_statistics(),
        }

    def _validate_nodes(self) -> None:
        """노드 유효성 검증"""
        node_ids = set()

        for node_id, node in self.graph.nodes.items():
            # ID 중복 검사
            if node.id != node_id:
                self.errors.append(f"노드 ID 불일치: {node_id} != {node.id}")

            if node.id in node_ids:
                self.errors.append(f"중복 노드 ID: {node.id}")
            node_ids.add(node.id)

            # 필수 필드 검사
            if not node.name:
                self.warnings.append(f"노드 이름이 비어있음: {node.id}")

            if not node.file_path:
                self.warnings.append(f"파일 경로가 비어있음: {node.id}")

    def _validate_relations(self) -> None:
        """관계 유효성 검증"""
        for i, relation in enumerate(self.graph.relations):
            # 노드 존재 확인
            if relation.from_node_id not in self.graph.nodes:
                self.errors.append(
                    f"관계 {i}: 시작 노드 없음 - {relation.from_node_id}"
                )

            if relation.to_node_id not in self.graph.nodes:
                # 외부 노드인지 확인
                if not relation.to_node_id.startswith("external:"):
                    self.errors.append(
                        f"관계 {i}: 대상 노드 없음 - {relation.to_node_id}"
                    )

            # 자기 참조 확인
            if relation.from_node_id == relation.to_node_id:
                self.warnings.append(f"관계 {i}: 자기 참조 - {relation.from_node_id}")

            # 가중치 유효성
            if relation.weight < 0:
                self.warnings.append(f"관계 {i}: 음수 가중치 - {relation.weight}")

    def _validate_consistency(self) -> None:
        """일관성 검증"""
        # 프로젝트 통계와 실제 데이터 비교
        actual_files = len({node.file_path for node in self.graph.nodes.values()})
        if self.graph.total_files != actual_files:
            self.warnings.append(
                f"파일 수 불일치: 통계={self.graph.total_files}, 실제={actual_files}"
            )

    def _get_validation_statistics(self) -> dict[str, Any]:
        """검증 통계 정보"""
        return {
            "total_nodes": len(self.graph.nodes),
            "total_relations": len(self.graph.relations),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "node_types": {
                node_type.value: len(self.graph.get_nodes_by_type(node_type))
                for node_type in NodeType
            },
            "relation_types": {
                rel_type.value: len(self.graph.get_relations_by_type(rel_type))
                for rel_type in RelationType
            },
        }


def validate_graph(graph: CodeGraph) -> dict[str, Any]:
    """그래프 유효성 검증 편의 함수"""
    validator = GraphValidator(graph)
    return validator.validate()

"""Graph 패키지 유틸리티 함수들

그래프 조작, 검증, 시각화 등의 편의 기능을 제공합니다.
- GraphValidator: 그래프 유효성 검증
- GraphExporter: JSON, DOT 형식으로 내보내기
- GraphAnalyzer: 그래프 분석 (순환 의존성, 복잡도 등)
"""

import json
import logging
from pathlib import Path
from typing import Any

from .models import CodeGraph, NodeType, RelationType

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

            # 라인 번호 유효성
            if node.start_line > node.end_line:
                self.errors.append(f"시작 라인이 종료 라인보다 큼: {node.id}")

            if node.start_line < 0 or node.end_line < 0:
                self.warnings.append(f"음수 라인 번호: {node.id}")

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

        # 라인 수 계산
        actual_lines = sum(
            max(node.end_line - node.start_line + 1, 0)
            for node in self.graph.nodes.values()
        )
        if abs(self.graph.total_lines - actual_lines) > 10:  # 10라인 오차 허용
            self.warnings.append(
                f"라인 수 차이: 통계={self.graph.total_lines}, 계산={actual_lines}"
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


class GraphExporter:
    """그래프 내보내기 클래스

    그래프를 JSON, DOT 등 다양한 형식으로 내보냅니다.
    """

    def __init__(self, graph: CodeGraph) -> None:
        """초기화

        Args:
            graph: 내보낼 CodeGraph
        """
        self.graph = graph
        logger.debug(f"GraphExporter initialized for graph: {graph.project_name}")

    def to_json(self, indent: int = 2) -> str:
        """JSON 형식으로 내보내기

        Args:
            indent: JSON 들여쓰기 크기

        Returns:
            JSON 문자열
        """
        data = {
            "project": {
                "name": self.graph.project_name,
                "path": self.graph.project_path,
                "total_files": self.graph.total_files,
                "total_lines": self.graph.total_lines,
                "analysis_version": self.graph.analysis_version,
                "created_at": self.graph.created_at.isoformat(),
                "updated_at": self.graph.updated_at.isoformat(),
            },
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type
                    if isinstance(node.node_type, str)
                    else node.node_type.value,
                    "file_path": node.file_path,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                    "source_code": node.source_code,
                    "docstring": node.docstring,
                    "complexity": node.complexity,
                    "scope_level": node.scope_level,
                    "parameters": node.parameters,
                    "return_type": node.return_type,
                    "decorators": node.decorators,
                    "imports": node.imports,
                    "dependencies": [
                        {
                            "target": dep.target,
                            "type": dep.dependency_type,
                            "line_number": dep.line_number,
                            "context": dep.context,
                        }
                        for dep in node.dependencies
                    ],
                }
                for node in self.graph.nodes.values()
            ],
            "relations": [
                {
                    "from": rel.from_node_id,
                    "to": rel.to_node_id,
                    "type": rel.relation_type
                    if isinstance(rel.relation_type, str)
                    else rel.relation_type.value,
                    "weight": rel.weight,
                    "line_number": rel.line_number,
                    "context": rel.context,
                    "created_at": rel.created_at.isoformat(),
                }
                for rel in self.graph.relations
            ],
            "statistics": self.graph.get_statistics(),
        }

        return json.dumps(data, indent=indent, ensure_ascii=False)

    def save_json(self, file_path: str, indent: int = 2):
        """JSON 파일로 저장"""
        json_data = self.to_json(indent)
        Path(file_path).write_text(json_data, encoding="utf-8")

    def to_dot(self) -> str:
        """DOT (Graphviz) 형식으로 내보내기"""
        lines = ["digraph CodeGraph {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=filled];")
        lines.append("")

        # 노드 타입별 색상 정의
        type_colors = {
            NodeType.MODULE: "lightblue",
            NodeType.CLASS: "lightgreen",
            NodeType.FUNCTION: "lightyellow",
            NodeType.METHOD: "lightcoral",
            NodeType.IMPORT: "lightgray",
            NodeType.VARIABLE: "lightpink",
            NodeType.PROPERTY: "lightcyan",
            NodeType.DECORATOR: "lightsalmon",
        }

        # 노드 정의
        for node in self.graph.nodes.values():
            color = type_colors.get(node.node_type, "white")
            node_type_value = (
                node.node_type
                if isinstance(node.node_type, str)
                else node.node_type.value
            )
            label = f"{node.name}\\n({node_type_value})"
            lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}"];')

        lines.append("")

        # 관계 정의
        for rel in self.graph.relations:
            if not rel.to_node_id.startswith("external:"):
                rel_type_value = (
                    rel.relation_type
                    if isinstance(rel.relation_type, str)
                    else rel.relation_type.value
                )
                lines.append(
                    f'  "{rel.from_node_id}" -> "{rel.to_node_id}" '
                    f'[label="{rel_type_value}"];'
                )

        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, file_path: str):
        """DOT 파일로 저장"""
        dot_data = self.to_dot()
        Path(file_path).write_text(dot_data, encoding="utf-8")


class GraphAnalyzer:
    """그래프 분석 유틸리티

    순환 의존성, 복잡도, 연결성 등을 분석합니다.
    """

    def __init__(self, graph: CodeGraph) -> None:
        """초기화

        Args:
            graph: 분석할 CodeGraph
        """
        self.graph = graph
        logger.debug(f"GraphAnalyzer initialized for graph: {graph.project_name}")

    def find_circular_dependencies(self) -> list[list[str]]:
        """순환 의존성 탐지"""
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node_id: str, path: list[str]):
            if node_id in rec_stack:
                # 순환 발견
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
                return

            if node_id in visited or node_id.startswith("external:"):
                return

            visited.add(node_id)
            rec_stack.add(node_id)

            # 이 노드에서 시작하는 모든 관계 확인
            for rel in self.graph.get_relations_from_node(node_id):
                dfs(rel.to_node_id, path + [node_id])

            rec_stack.remove(node_id)

        # 모든 노드에서 DFS 시작
        for node_id in self.graph.nodes.keys():
            if node_id not in visited:
                dfs(node_id, [])

        return cycles

    def get_dependency_depth(self, node_id: str) -> int:
        """노드의 의존성 깊이 계산"""
        if node_id not in self.graph.nodes:
            return 0

        visited = set()

        def dfs(current_id: str) -> int:
            if current_id in visited or current_id.startswith("external:"):
                return 0

            visited.add(current_id)

            max_depth = 0
            for rel in self.graph.get_relations_from_node(current_id):
                if rel.relation_type in [RelationType.CALLS, RelationType.DEPENDS_ON]:
                    depth = dfs(rel.to_node_id)
                    max_depth = max(max_depth, depth)

            return max_depth + 1

        return dfs(node_id)

    def find_most_connected_nodes(self, top_n: int = 10) -> list[dict[str, Any]]:
        """가장 연결된 노드들 찾기"""
        connection_counts = {}

        for node_id in self.graph.nodes.keys():
            incoming = len(self.graph.get_relations_to_node(node_id))
            outgoing = len(self.graph.get_relations_from_node(node_id))
            connection_counts[node_id] = {
                "node": self.graph.nodes[node_id],
                "incoming": incoming,
                "outgoing": outgoing,
                "total": incoming + outgoing,
            }

        # 총 연결 수로 정렬
        sorted_nodes = sorted(
            connection_counts.items(), key=lambda x: x[1]["total"], reverse=True
        )

        return [
            {
                "node_id": node_id,
                "name": data["node"].name,
                "type": data["node"].node_type
                if isinstance(data["node"].node_type, str)
                else data["node"].node_type.value,
                "file_path": data["node"].file_path,
                "incoming_connections": data["incoming"],
                "outgoing_connections": data["outgoing"],
                "total_connections": data["total"],
            }
            for node_id, data in sorted_nodes[:top_n]
        ]

    def get_file_complexity_ranking(self) -> list[dict[str, Any]]:
        """파일별 복잡도 순위"""
        file_complexity = {}

        for node in self.graph.nodes.values():
            file_path = node.file_path
            if file_path not in file_complexity:
                file_complexity[file_path] = {
                    "total_complexity": 0,
                    "node_count": 0,
                    "total_lines": 0,
                }

            file_complexity[file_path]["total_complexity"] += node.complexity
            file_complexity[file_path]["node_count"] += 1
            file_complexity[file_path]["total_lines"] += (
                node.end_line - node.start_line + 1
            )

        # 평균 복잡도로 정렬
        ranked_files = []
        for file_path, stats in file_complexity.items():
            avg_complexity = (
                stats["total_complexity"] / stats["node_count"]
                if stats["node_count"] > 0
                else 0
            )
            ranked_files.append(
                {
                    "file_path": file_path,
                    "total_complexity": stats["total_complexity"],
                    "average_complexity": avg_complexity,
                    "node_count": stats["node_count"],
                    "total_lines": stats["total_lines"],
                }
            )

        return sorted(ranked_files, key=lambda x: x["average_complexity"], reverse=True)


# 편의 함수들
def validate_graph(graph: CodeGraph) -> dict[str, Any]:
    """그래프 유효성 검증 편의 함수"""
    validator = GraphValidator(graph)
    return validator.validate()


def export_graph_json(graph: CodeGraph, file_path: str, indent: int = 2):
    """그래프를 JSON으로 내보내기 편의 함수"""
    exporter = GraphExporter(graph)
    exporter.save_json(file_path, indent)


def export_graph_dot(graph: CodeGraph, file_path: str):
    """그래프를 DOT으로 내보내기 편의 함수"""
    exporter = GraphExporter(graph)
    exporter.save_dot(file_path)


def analyze_graph_complexity(graph: CodeGraph) -> dict[str, Any]:
    """그래프 복잡도 분석 편의 함수"""
    analyzer = GraphAnalyzer(graph)

    return {
        "circular_dependencies": analyzer.find_circular_dependencies(),
        "most_connected_nodes": analyzer.find_most_connected_nodes(5),
        "file_complexity_ranking": analyzer.get_file_complexity_ranking(),
    }

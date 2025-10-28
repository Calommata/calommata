import logging
from typing import Any

from src.graph.db import CodeGraph, RelationType

logger = logging.getLogger(__name__)


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
                "type": (
                    data["node"].node_type.value
                    if hasattr(data["node"].node_type, "value")
                    else data["node"].node_type
                ),
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
                }

            file_complexity[file_path]["total_complexity"] += node.complexity
            file_complexity[file_path]["node_count"] += 1

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
                }
            )

        return sorted(ranked_files, key=lambda x: x["average_complexity"], reverse=True)


def analyze_graph_complexity(graph: CodeGraph) -> dict[str, Any]:
    """그래프 복잡도 분석 편의 함수"""
    analyzer = GraphAnalyzer(graph)

    return {
        "circular_dependencies": analyzer.find_circular_dependencies(),
        "most_connected_nodes": analyzer.find_most_connected_nodes(5),
        "file_complexity_ranking": analyzer.get_file_complexity_ranking(),
    }

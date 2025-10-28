import json
import logging
from pathlib import Path

from src.graph.db import CodeGraph, NodeType

logger = logging.getLogger(__name__)


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
                    "type": (
                        node.node_type.value
                        if hasattr(node.node_type, "value")
                        else node.node_type
                    ),
                    "file_path": node.file_path,
                    "source_code": node.source_code,
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
                    "type": (
                        rel.relation_type.value
                        if hasattr(rel.relation_type, "value")
                        else rel.relation_type
                    ),
                    "weight": rel.weight,
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
            node_type_str = (
                node.node_type.value
                if hasattr(node.node_type, "value")
                else node.node_type
            )
            label = f"{node.name}\\n({node_type_str})"
            lines.append(f'  "{node.id}" [label="{label}", fillcolor="{color}"];')

        lines.append("")

        # 관계 정의
        for rel in self.graph.relations:
            if not rel.to_node_id.startswith("external:"):
                rel_type_str = (
                    rel.relation_type.value
                    if hasattr(rel.relation_type, "value")
                    else rel.relation_type
                )
                lines.append(
                    f'  "{rel.from_node_id}" -> "{rel.to_node_id}" '
                    f'[label="{rel_type_str}"];'
                )

        lines.append("}")
        return "\n".join(lines)

    def save_dot(self, file_path: str):
        """DOT 파일로 저장"""
        dot_data = self.to_dot()
        Path(file_path).write_text(dot_data, encoding="utf-8")


def export_graph_to_json(graph: CodeGraph, file_path: str, indent: int = 2):
    """그래프를 JSON으로 내보내기 편의 함수"""
    exporter = GraphExporter(graph)
    exporter.save_json(file_path, indent)


def export_graph_to_dot(graph: CodeGraph, file_path: str):
    """그래프를 DOT으로 내보내기 편의 함수"""
    exporter = GraphExporter(graph)
    exporter.save_dot(file_path)

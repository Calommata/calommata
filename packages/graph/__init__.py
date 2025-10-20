"""
Code Analyzer Graph Package

코드 분석 그래프 데이터 구조 및 변환 도구
Parser 결과를 구조화된 그래프 모델로 변환
"""

from .src.adapter import (
    ParserToGraphAdapter,
    convert_parser_results_to_graph,
    create_node_from_dict,
)
from .src.models import (
    CodeGraph,
    CodeNode,
    CodeRelation,
    Dependency,
    NodeType,
    RelationType,
)
from .src.utils import (
    GraphAnalyzer,
    GraphExporter,
    GraphValidator,
    analyze_graph_complexity,
    export_graph_dot,
    export_graph_json,
    validate_graph,
)

__version__ = "0.2.0"
__author__ = "Code Analyzer Team"

__all__ = [
    # 데이터 모델
    "CodeGraph",
    "CodeNode",
    "CodeRelation",
    "NodeType",
    "RelationType",
    "Dependency",
    # 어댑터
    "ParserToGraphAdapter",
    "convert_parser_results_to_graph",
    "create_node_from_dict",
    # 유틸리티
    "GraphValidator",
    "GraphExporter",
    "GraphAnalyzer",
    "validate_graph",
    "export_graph_json",
    "export_graph_dot",
    "analyze_graph_complexity",
]

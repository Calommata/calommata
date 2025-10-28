"""간소화된 Parser-to-Graph 어댑터

단일 책임 원칙을 적용하여 변환 작업을 전문 클래스들에 위임합니다.
"""

import logging
from typing import Any

from .models import CodeGraph
from .node_converter import NodeConverter
from .relationship_builder import RelationshipBuilder
from .statistics_updater import GraphStatisticsUpdater

logger = logging.getLogger(__name__)


class ParserToGraphAdapter:
    """Parser 결과를 Graph 모델로 변환하는 간소화된 어댑터

    실제 변환 작업은 전문 클래스들에 위임:
    - NodeConverter: 노드 변환
    - RelationshipBuilder: 관계 생성
    """

    def __init__(self):
        self.node_converter = NodeConverter()
        self.relationship_builder = RelationshipBuilder()

    def convert_to_graph(
        self,
        parser_results: list[dict[str, Any]] | list[Any],
        project_name: str = "unknown",
        project_path: str = "unknown",
    ) -> CodeGraph:
        """Parser 결과를 CodeGraph로 변환

        Args:
            parser_results: Parser 결과 (CodeBlock 리스트 또는 dict 리스트)
            project_name: 프로젝트 이름
            project_path: 프로젝트 경로

        Returns:
            변환된 CodeGraph
        """
        graph = CodeGraph(project_name=project_name, project_path=project_path)

        if not parser_results:
            logger.warning("변환할 parser 결과가 없습니다")
            return graph

        # CodeBlock 객체인지 dict인지 판단
        if hasattr(parser_results[0], "block_type"):
            return self._convert_from_code_blocks(parser_results, graph)
        else:
            return self._convert_from_dicts(parser_results, graph)

    def _convert_from_code_blocks(
        self,
        code_blocks: list[Any],
        graph: CodeGraph,
    ) -> CodeGraph:
        """CodeBlock 객체들을 Graph로 변환

        Args:
            code_blocks: CodeBlock 리스트
            graph: 대상 CodeGraph

        Returns:
            변환 완료된 CodeGraph
        """
        node_map = {}  # key: "file_path:name" -> value: node_id

        # 1단계: 노드 변환 및 추가
        for block in code_blocks:
            node = self.node_converter.convert_block_to_node(block)
            graph.add_node(node)

            file_path = getattr(block, "file_path", "unknown.py")
            node_map[f"{file_path}:{block.name}"] = node.id

        # 2단계: 의존성 기반 관계 생성
        self.relationship_builder.build_dependency_relations(
            code_blocks, graph, node_map
        )

        # 3단계: 구조적 관계 생성
        self.relationship_builder.build_structural_relations(
            code_blocks, graph, node_map
        )

        # 4단계: 통계 업데이트
        GraphStatisticsUpdater.update_statistics(graph)

        logger.info(
            f"✅ Graph 변환 완료: {len(graph.nodes)}개 노드, "
            f"{len(graph.relations)}개 관계"
        )
        return graph

    def _convert_from_dicts(
        self,
        parser_results: list[dict[str, Any]],
        graph: CodeGraph,
    ) -> CodeGraph:
        """딕셔너리 데이터를 Graph로 변환

        Args:
            parser_results: 딕셔너리 리스트
            graph: 대상 CodeGraph

        Returns:
            변환 완료된 CodeGraph
        """
        node_map = {}  # key: full_name -> value: node_id

        # 1단계: 노드 변환 및 추가
        for block_data in parser_results:
            node = self.node_converter.convert_dict_to_node(block_data)
            graph.add_node(node)

            full_name = block_data.get("full_name", f"{node.file_path}:{node.name}")
            node_map[full_name] = node.id

        # 2단계: 관계 생성
        for block_data in parser_results:
            relations = self.relationship_builder.build_relations_from_dict(
                block_data, node_map
            )
            for relation in relations:
                try:
                    graph.add_relation(relation)
                except ValueError:
                    continue

        # 3단계: 통계 업데이트
        GraphStatisticsUpdater.update_statistics(graph)

        logger.info(
            f"✅ Graph 변환 완료: {len(graph.nodes)}개 노드, "
            f"{len(graph.relations)}개 관계"
        )
        return graph

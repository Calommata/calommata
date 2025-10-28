"""관계 생성 오케스트레이터

관계 생성을 전문 클래스들에 위임합니다.
"""

import logging
from typing import Any

from .dict_relation_builder import DictRelationBuilder
from .dependency_relation_builder import DependencyRelationBuilder
from .models import CodeGraph, CodeRelation
from .structural_relation_builder import StructuralRelationBuilder

logger = logging.getLogger(__name__)


class RelationshipBuilder:
    """노드 간 관계 생성 오케스트레이터

    실제 관계 생성은 전문 클래스들에 위임:
    - DependencyRelationBuilder: 의존성 기반 관계
    - StructuralRelationBuilder: 구조적 관계 (CONTAINS)
    - DictRelationBuilder: dict 기반 관계
    """

    def __init__(self):
        self.dependency_builder = DependencyRelationBuilder()
        self.structural_builder = StructuralRelationBuilder()
        self.dict_builder = DictRelationBuilder()

    def build_dependency_relations(
        self,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
    ) -> None:
        """의존성 기반 관계 생성"""
        self.dependency_builder.build(blocks, graph, node_map)

    def build_structural_relations(
        self,
        blocks: list[Any],
        graph: CodeGraph,
        node_map: dict[str, str],
    ) -> None:
        """구조적 관계 생성 (CONTAINS)"""
        self.structural_builder.build(blocks, graph, node_map)

    def build_relations_from_dict(
        self,
        block_data: dict[str, Any],
        node_map: dict[str, str],
    ) -> list[CodeRelation]:
        """딕셔너리에서 관계 추출"""
        return self.dict_builder.build(block_data, node_map)

"""그래프 서비스 모듈 (오케스트레이터)

프로젝트 분석, 임베딩 생성, 통계 조회를 조합하여 전체 파이프라인을 관리합니다.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.graph import Neo4jPersistence, CodeGraph

from src.core.embedding import CodeEmbedder, GraphEmbedder
from src.core.utils import ProjectAnalyzer
from src.core.graph.graph_statistics import GraphStatistics

logger = logging.getLogger(__name__)


class CodeGraphService(BaseModel):
    """코드 그래프 서비스 오케스트레이터

    프로젝트 분석 -> 임베딩 생성 -> Neo4j 저장
    전체 파이프라인을 조합합니다.
    """

    persistence: Neo4jPersistence = Field(..., description="Neo4j 지속성 객체")
    embedder: CodeEmbedder = Field(..., description="코드 임베딩 생성기")
    project_name: str = Field(..., description="프로젝트 이름")

    _analyzer: ProjectAnalyzer | None = None
    _graph_embedder: GraphEmbedder | None = None
    _statistics: GraphStatistics | None = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def analyzer(self) -> ProjectAnalyzer:
        """프로젝트 분석기"""
        if self._analyzer is None:
            self._analyzer = ProjectAnalyzer(project_name=self.project_name)
        return self._analyzer

    @property
    def graph_embedder(self) -> GraphEmbedder:
        """그래프 임베딩 생성기"""
        if self._graph_embedder is None:
            self._graph_embedder = GraphEmbedder(
                embedder=self.embedder,
                persistence=self.persistence,
            )
        return self._graph_embedder

    @property
    def statistics(self) -> GraphStatistics:
        """그래프 통계"""
        if self._statistics is None:
            self._statistics = GraphStatistics(
                persistence=self.persistence,
                project_name=self.project_name,
            )
        return self._statistics

    def analyze_and_store_project(
        self,
        project_path: str,
        create_embeddings: bool = True,
    ) -> CodeGraph:
        """프로젝트 분석 및 그래프 저장

        Args:
            project_path: 분석할 프로젝트 경로
            create_embeddings: 임베딩 생성 여부

        Returns:
            생성된 CodeGraph
        """
        # 1. 분석
        graph = self.analyzer.analyze_and_store_project(project_path)

        # 2. 임베딩 생성
        if create_embeddings:
            self.graph_embedder.create_embeddings_for_graph(graph)

            # 벡터 인덱스 최적화
            if graph.nodes:
                first_node = next(iter(graph.nodes.values()))
                if first_node.embedding_vector:
                    embedding_dim = len(first_node.embedding_vector)
                    self.persistence.create_vector_index_for_dimension(embedding_dim)
                    logger.info(f"✅ {embedding_dim}차원 벡터 인덱스 최적화 완료")

        # 3. Neo4j에 저장
        self.persistence.save_code_graph(graph, self.project_name)
        logger.info("✅ Neo4j 저장 완료")

        return graph

    def analyze_and_store_file(
        self,
        file_path: str,
        create_embeddings: bool = True,
    ) -> CodeGraph:
        """단일 파일 분석 및 그래프 저장

        Args:
            file_path: 분석할 파일 경로
            create_embeddings: 임베딩 생성 여부

        Returns:
            생성된 CodeGraph
        """
        # 1. 분석
        graph = self.analyzer.analyze_and_store_file(file_path)

        # 2. 임베딩 생성
        if create_embeddings:
            self.graph_embedder.create_embeddings_for_graph(graph)

        # 3. Neo4j에 저장
        self.persistence.save_code_graph(graph, self.project_name)
        logger.info("✅ Neo4j 저장 완료")

        return graph

    def update_embeddings(self, node_ids: list[str] | None = None) -> None:
        """특정 노드들의 임베딩 업데이트

        Args:
            node_ids: 업데이트할 노드 ID 리스트
        """
        self.graph_embedder.update_embeddings(node_ids)

    def get_statistics(self) -> dict[str, Any]:
        """프로젝트 통계 정보 조회

        Returns:
            통계 정보 딕셔너리
        """
        return self.statistics.get_statistics()

    def clear_project(self) -> bool:
        """프로젝트 데이터 삭제

        Returns:
            성공 여부
        """
        return self.statistics.clear_project()

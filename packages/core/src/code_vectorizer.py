"""
코드 벡터화 서비스
코드 블록들을 임베딩으로 변환하고 Neo4j에 저장

Neo4jPersistence를 사용하여 데이터베이스 작업을 수행합니다.
"""

import logging
from typing import Any

from .embedding_service import EmbeddingService
from graph.src.persistence import Neo4jPersistence


class CodeVectorizer:
    """코드 블록 벡터화 및 저장 관리"""

    def __init__(
        self,
        neo4j_persistence,
        embedding_service=None,
    ):
        """벡터화 서비스 초기화

        Args:
            neo4j_persistence: Neo4j 지속성 계층
            embedding_service: 임베딩 서비스 (None이면 기본값 생성)
        """
        self.neo4j_persistence = neo4j_persistence
        self.embedding_service = embedding_service or EmbeddingService()
        self.logger = logging.getLogger(__name__)

    def vectorize_project_nodes(
        self, project_name: str, force_update: bool = False
    ) -> bool:
        """프로젝트의 모든 노드를 벡터화

        Args:
            project_name: 프로젝트명
            force_update: True면 기존 임베딩 무시하고 다시 생성

        Returns:
            bool: 벡터화 성공 여부
        """
        if not self.neo4j_persistence:
            self.logger.error("Neo4j 지속성 계층이 없습니다")
            return False

        try:
            # 벡터화 통계 조회
            stats = self._get_vectorization_needs(project_name, force_update)

            if not stats or stats.get("total_nodes", 0) == 0:
                self.logger.info("벡터화할 노드가 없습니다")
                return True

            nodes_to_vectorize = stats.get("total_nodes", 0)
            self.logger.info(f"벡터화할 노드 수: {nodes_to_vectorize}")

            if force_update:
                self.logger.info("기존 임베딩을 무시하고 모든 노드를 벡터화합니다")

            # 임베딩 서비스 정보
            embedding_info = self.embedding_service.get_embedding_info()
            self.logger.info(
                f"임베딩 모델: {embedding_info.get('model_name')} "
                f"({embedding_info.get('dimensions')}차원)"
            )

            self.logger.info(
                f"벡터화 준비 완료: {nodes_to_vectorize}개 노드 "
                f"({self.embedding_service.config.batch_size}배치 크기)"
            )

            return True

        except Exception as e:
            self.logger.error(f"프로젝트 벡터화 준비 실패: {e}")
            return False

    def _get_vectorization_needs(
        self, project_name: str, force_update: bool
    ) -> dict[str, Any]:
        """벡터화 필요 현황 조회

        Args:
            project_name: 프로젝트명
            force_update: True면 모든 노드, False면 임베딩 없는 노드만

        Returns:
            dict: 벡터화 통계
        """
        try:
            if not self.neo4j_persistence or not self.neo4j_persistence.driver:
                return {}

            with self.neo4j_persistence.driver.session() as session:
                if force_update:
                    # 모든 노드 조회
                    query = """
                    MATCH (p:Project {name: $project_name})-[:CONTAINS]->(n:CodeNode)
                    RETURN count(n) AS total_nodes,
                           count(DISTINCT n.id) AS unique_nodes
                    """
                else:
                    # 임베딩이 없는 노드만 조회
                    query = """
                    MATCH (p:Project {name: $project_name})-[:CONTAINS]->(n:CodeNode)
                    WHERE n.embedding IS NULL
                    RETURN count(n) AS total_nodes,
                           count(DISTINCT n.id) AS unique_nodes
                    """

                result = session.run(query, project_name=project_name)
                record = result.single()

                if record:
                    return {
                        "total_nodes": record["total_nodes"],
                        "unique_nodes": record["unique_nodes"],
                    }
                else:
                    return {"total_nodes": 0, "unique_nodes": 0}

        except Exception as e:
            self.logger.error(f"벡터화 필요 현황 조회 실패: {e}")
            return {}

    def get_vectorization_statistics(self, project_name: str) -> dict[str, Any]:
        """벡터화 통계 정보 조회

        Args:
            project_name: 프로젝트명

        Returns:
            dict: 벡터화 통계 (총 노드, 벡터화된 노드, 진행률 등)
        """
        try:
            if not self.neo4j_persistence or not self.neo4j_persistence.driver:
                return {
                    "total_nodes": 0,
                    "vectorized_nodes": 0,
                    "remaining_nodes": 0,
                    "vectorization_progress": 0,
                    "models_used": [],
                    "embedding_service": self.embedding_service.get_embedding_info(),
                }

            with self.neo4j_persistence.driver.session() as session:
                query = """
                MATCH (p:Project {name: $project_name})-[:CONTAINS]->(n:CodeNode)
                RETURN count(n) AS total_nodes,
                       count(n.embedding) AS vectorized_nodes,
                       collect(DISTINCT n.embedding_model) AS models_used
                """

                result = session.run(query, project_name=project_name)
                record = result.single()

                if record:
                    total = record["total_nodes"]
                    vectorized = record["vectorized_nodes"]

                    return {
                        "total_nodes": total,
                        "vectorized_nodes": vectorized,
                        "remaining_nodes": total - vectorized,
                        "vectorization_progress": (vectorized / total) * 100
                        if total > 0
                        else 0,
                        "models_used": [m for m in record["models_used"] if m],
                        "embedding_service": self.embedding_service.get_embedding_info(),
                    }
                else:
                    return {
                        "total_nodes": 0,
                        "vectorized_nodes": 0,
                        "remaining_nodes": 0,
                        "vectorization_progress": 0,
                        "models_used": [],
                        "embedding_service": self.embedding_service.get_embedding_info(),
                    }

        except Exception as e:
            self.logger.error(f"벡터화 통계 조회 실패: {e}")
            return {}

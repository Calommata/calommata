"""
코드 벡터화 서비스
코드 블록들을 임베딩으로 변환하고 Neo4j에 저장
"""

import logging
from typing import Optional, Any

from .embedding_service import EmbeddingService
from .neo4j_handler import Neo4jHandler


class CodeVectorizer:
    """코드 블록 벡터화 및 저장 관리"""

    def __init__(
        self,
        neo4j_handler: Neo4jHandler,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.neo4j_handler = neo4j_handler
        self.embedding_service = embedding_service or EmbeddingService()
        self.logger = logging.getLogger(__name__)

    def vectorize_project_nodes(
        self, project_name: str, force_update: bool = False
    ) -> bool:
        """프로젝트의 모든 노드를 벡터화"""
        try:
            # 임베딩이 없는 노드들 조회
            nodes_to_vectorize = self._get_nodes_without_embedding(
                project_name, force_update
            )

            if not nodes_to_vectorize:
                self.logger.info("벡터화할 노드가 없습니다")
                return True

            self.logger.info(f"벡터화할 노드 수: {len(nodes_to_vectorize)}")

            # 배치로 임베딩 생성
            success_count = 0

            for node in nodes_to_vectorize:
                try:
                    # 임베딩 생성
                    embedding = self.embedding_service.create_code_embedding(
                        source_code=node.get("source_code", ""),
                        docstring=node.get("docstring", ""),
                    )

                    if embedding:
                        # Neo4j에 임베딩 저장
                        success = self.neo4j_handler.update_node_embedding(
                            node_id=node["id"],
                            embedding=embedding,
                            model=self.embedding_service.config.model_name,
                        )

                        if success:
                            success_count += 1
                        else:
                            self.logger.warning(f"임베딩 저장 실패: {node['id']}")
                    else:
                        self.logger.warning(f"임베딩 생성 실패: {node['id']}")

                except Exception as e:
                    self.logger.error(f"노드 벡터화 실패 {node['id']}: {e}")
                    continue

            self.logger.info(f"벡터화 완료: {success_count}/{len(nodes_to_vectorize)}")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"프로젝트 벡터화 실패: {e}")
            return False

    def _get_nodes_without_embedding(
        self, project_name: str, force_update: bool
    ) -> list[dict[str, Any]]:
        """임베딩이 없는 노드들 조회"""
        try:
            with self.neo4j_handler.driver.session() as session:
                if force_update:
                    # 모든 노드 조회
                    query = """
                    MATCH (p:Project {name: $project_name})-[:CONTAINS]->(n:CodeNode)
                    RETURN n.id AS id,
                           n.source_code AS source_code,
                           n.docstring AS docstring,
                           n.type AS type,
                           n.name AS name
                    """
                else:
                    # 임베딩이 없는 노드만 조회
                    query = """
                    MATCH (p:Project {name: $project_name})-[:CONTAINS]->(n:CodeNode)
                    WHERE n.embedding IS NULL
                    RETURN n.id AS id,
                           n.source_code AS source_code,
                           n.docstring AS docstring,
                           n.type AS type,
                           n.name AS name
                    """

                result = session.run(query, project_name=project_name)
                return [record.data() for record in result]

        except Exception as e:
            self.logger.error(f"노드 조회 실패: {e}")
            return []

    def vectorize_single_node(self, node_id: str) -> bool:
        """단일 노드 벡터화"""
        try:
            # 노드 정보 조회
            with self.neo4j_handler.driver.session() as session:
                query = """
                MATCH (n:CodeNode {id: $node_id})
                RETURN n.source_code AS source_code,
                       n.docstring AS docstring
                """

                result = session.run(query, node_id=node_id)
                record = result.single()

                if not record:
                    self.logger.error(f"노드를 찾을 수 없음: {node_id}")
                    return False

                # 임베딩 생성
                embedding = self.embedding_service.create_code_embedding(
                    source_code=record["source_code"], docstring=record["docstring"]
                )

                if embedding:
                    # Neo4j에 저장
                    return self.neo4j_handler.update_node_embedding(
                        node_id=node_id,
                        embedding=embedding,
                        model=self.embedding_service.config.model_name,
                    )
                else:
                    return False

        except Exception as e:
            self.logger.error(f"단일 노드 벡터화 실패: {e}")
            return False

    def get_vectorization_statistics(self, project_name: str) -> dict[str, Any]:
        """벡터화 통계 정보"""
        try:
            with self.neo4j_handler.driver.session() as session:
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

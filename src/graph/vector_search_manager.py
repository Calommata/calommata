"""벡터 검색 관리 모듈

벡터 유사도 기반 검색을 담당합니다.
"""

import logging
from typing import Any

from neo4j import Driver

from src.graph.db import Neo4jQueries

logger = logging.getLogger(__name__)


class VectorSearchManager:
    """벡터 검색 관리"""

    def __init__(self, driver: Driver):
        """초기화

        Args:
            driver: Neo4j 드라이버
        """
        self.driver = driver
        self._queries = Neo4jQueries()

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        project_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """벡터 유사도 기반 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            limit: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)
            project_name: 프로젝트명 (선택적)

        Returns:
            list[dict]: 유사한 노드들의 정보 리스트
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    self._queries.VECTOR_SEARCH,
                    query_embedding=query_embedding,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    project_name=project_name,
                )

                results = [record.data() for record in result]
                logger.info(f"✅ 벡터 검색 완료: {len(results)}개 결과")
                return results

        except Exception as e:
            logger.error(f"❌ 벡터 검색 실패: {e}")
            return []

    def get_node_context(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        """노드와 주변 컨텍스트 조회

        Args:
            node_id: 조회할 노드 ID
            depth: 주변 관계 깊이 (기본값: 2)

        Returns:
            dict: 노드 정보와 관련 노드들
        """
        try:
            with self.driver.session() as session:
                # 깊이에 따른 쿼리 선택
                if depth == 1:
                    query = """
                        MATCH (center:CodeNode {id: $node_id})
                        OPTIONAL MATCH path = (center)-[*1..1]-(related:CodeNode)
                        WITH center, collect(DISTINCT related) AS related_nodes,
                             [rel IN collect(DISTINCT relationships(path)) WHERE rel IS NOT NULL | rel] AS all_relationship_lists
                        
                        RETURN center,
                               related_nodes,
                               [rel IN reduce(flat = [], list IN all_relationship_lists | flat + list) WHERE rel IS NOT NULL | {
                                   type: type(rel),
                                   start_node: startNode(rel).id,
                                   end_node: endNode(rel).id,
                                   properties: properties(rel)
                               }] AS relationships
                    """
                elif depth == 3:
                    query = """
                        MATCH (center:CodeNode {id: $node_id})
                        OPTIONAL MATCH path = (center)-[*1..3]-(related:CodeNode)
                        WITH center, collect(DISTINCT related) AS related_nodes,
                             [rel IN collect(DISTINCT relationships(path)) WHERE rel IS NOT NULL | rel] AS all_relationship_lists
                        
                        RETURN center,
                               related_nodes,
                               [rel IN reduce(flat = [], list IN all_relationship_lists | flat + list) WHERE rel IS NOT NULL | {
                                   type: type(rel),
                                   start_node: startNode(rel).id,
                                   end_node: endNode(rel).id,
                                   properties: properties(rel)
                               }] AS relationships
                    """
                else:  # depth == 2 (기본값)
                    query = """
                        MATCH (center:CodeNode {id: $node_id})
                        OPTIONAL MATCH path = (center)-[*1..2]-(related:CodeNode)
                        WITH center, collect(DISTINCT related) AS related_nodes,
                             [rel IN collect(DISTINCT relationships(path)) WHERE rel IS NOT NULL | rel] AS all_relationship_lists
                        
                        RETURN center,
                               related_nodes,
                               [rel IN reduce(flat = [], list IN all_relationship_lists | flat + list) WHERE rel IS NOT NULL | {
                                   type: type(rel),
                                   start_node: startNode(rel).id,
                                   end_node: endNode(rel).id,
                                   properties: properties(rel)    
                               }] AS relationships
                    """

                result = session.run(query, node_id=node_id)  # type: ignore[arg-type]
                record = result.single()

                if record:
                    # 관계를 필터링하여 None 제거
                    relationships = [
                        rel for rel in record["relationships"] if rel is not None
                    ]

                    return {
                        "center_node": dict(record["center"]),
                        "related_nodes": [
                            dict(node) for node in record["related_nodes"]
                        ],
                        "relationships": relationships,
                    }
                else:
                    logger.warning(f"⚠️ 노드를 찾을 수 없음: {node_id}")
                    return {
                        "center_node": None,
                        "related_nodes": [],
                        "relationships": [],
                    }

        except Exception as e:
            logger.error(f"❌ 노드 컨텍스트 조회 실패: {e}")
            return {"center_node": None, "related_nodes": [], "relationships": []}

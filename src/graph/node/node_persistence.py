"""코드 노드 지속성 모듈

노드 저장, 조회, 임베딩 업데이트를 담당합니다.
"""

import logging

from neo4j import Driver

from src.graph.db import CodeGraph, CodeNode, Neo4jQueries
from src.graph.node.error import NodeNotFoundError

logger = logging.getLogger(__name__)


class NodePersistence:
    """코드 노드 저장 및 조회"""

    DEFAULT_BATCH_SIZE = 500

    def __init__(self, driver: Driver, batch_size: int = DEFAULT_BATCH_SIZE):
        """초기화

        Args:
            driver: Neo4j 드라이버
            batch_size: 배치 크기
        """
        self.driver = driver
        self.batch_size = batch_size
        self._queries = Neo4jQueries()

    def save_project_info(self, graph: CodeGraph, project_name: str) -> None:
        """프로젝트 정보 저장

        Args:
            graph: CodeGraph 객체
            project_name: 프로젝트 이름
        """
        try:
            with self.driver.session() as session:
                stats = graph.get_statistics()
                session.run(
                    self._queries.MERGE_PROJECT,
                    name=project_name,
                    path=graph.project_path,
                    total_files=graph.total_files,
                    total_lines=graph.total_lines,
                    total_nodes=stats["total_nodes"],
                    total_relations=stats["total_relations"],
                    analysis_version=graph.analysis_version,
                    created_at=graph.created_at.isoformat(),
                )
                logger.info(f"✅ 프로젝트 정보 저장: {project_name}")

        except Exception as e:
            logger.error(f"❌ 프로젝트 정보 저장 실패: {e}")
            raise

    def save_nodes_batch(self, nodes: list[CodeNode], project_name: str) -> None:
        """코드 노드들을 배치로 저장

        Args:
            nodes: 저장할 노드 리스트
            project_name: 프로젝트 이름
        """
        if not nodes:
            logger.warning("⚠️ 저장할 노드가 없습니다")
            return

        try:
            total_batches = (len(nodes) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(nodes))
                batch = nodes[start_idx:end_idx]

                # Neo4j 형식으로 변환
                neo4j_nodes = [node.to_neo4j_node() for node in batch]

                with self.driver.session() as session:
                    session.run(
                        self._queries.MERGE_CODE_NODES_BATCH,
                        nodes=neo4j_nodes,
                        project_name=project_name,
                    )

                logger.info(
                    f"✅ 노드 배치 {batch_idx + 1}/{total_batches} 저장 완료 ({len(batch)}개)"
                )

            logger.info(f"✅ 총 {len(nodes)}개 코드 노드 저장 완료")

        except Exception as e:
            logger.error(f"❌ 코드 노드 배치 저장 실패: {e}")
            raise

    def update_embedding(
        self, node_id: str, embedding: list[float], model: str
    ) -> bool:
        """노드의 임베딩 벡터 업데이트

        Args:
            node_id: 업데이트할 노드 ID
            embedding: 임베딩 벡터
            model: 임베딩 모델명

        Returns:
            bool: 성공 여부

        Raises:
            NodeNotFoundError: 노드를 찾을 수 없을 때
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    self._queries.UPDATE_NODE_EMBEDDING,
                    node_id=node_id,
                    embedding=embedding,
                    model=model,
                )

                if result.single():
                    logger.info(f"✅ 임베딩 업데이트: {node_id}")
                    return True
                else:
                    raise NodeNotFoundError(f"노드를 찾을 수 없음: {node_id}")

        except NodeNotFoundError:
            raise
        except Exception as e:
            logger.error(f"❌ 임베딩 업데이트 실패: {e}")
            raise

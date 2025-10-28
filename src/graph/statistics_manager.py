"""통계 및 프로젝트 관리 모듈

데이터베이스 통계 조회 및 프로젝트 관리를 담당합니다.
"""

import logging
from typing import Any

from neo4j import Driver

from .queries import Neo4jQueries

logger = logging.getLogger(__name__)


class StatisticsManager:
    """데이터베이스 통계 조회 및 프로젝트 관리"""

    def __init__(self, driver: Driver):
        """초기화

        Args:
            driver: Neo4j 드라이버
        """
        self.driver = driver
        self._queries = Neo4jQueries()

    def get_statistics(self) -> dict[str, Any]:
        """데이터베이스 통계 정보 조회

        Returns:
            dict: 노드, 관계, 타입별 통계
        """
        try:
            with self.driver.session() as session:
                # 노드 수 조회
                node_stats = session.run(self._queries.GET_NODE_STATS).data()

                # 관계 수 조회
                rel_stats = session.run(self._queries.GET_RELATION_STATS).data()

                # 전체 통계
                total_stats_record = session.run(self._queries.GET_TOTAL_STATS).single()

                if not total_stats_record:
                    logger.warning("⚠️ 통계 데이터를 찾을 수 없음")
                    return {}

                return {
                    "total_nodes": total_stats_record["total_nodes"] or 0,
                    "total_relationships": total_stats_record["total_relationships"]
                    or 0,
                    "node_types": {
                        stat["type"]: stat["count"]
                        for stat in node_stats
                        if stat["type"]
                    },
                    "relation_types": {
                        stat["type"]: stat["count"]
                        for stat in rel_stats
                        if stat["type"]
                    },
                }

        except Exception as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return {}

    def clear_project_data(self, project_name: str) -> bool:
        """프로젝트 데이터 삭제

        Args:
            project_name: 삭제할 프로젝트명

        Returns:
            bool: 성공 여부
        """
        try:
            with self.driver.session() as session:
                session.run(self._queries.DELETE_PROJECT, project_name=project_name)
                logger.info(f"✅ 프로젝트 데이터 삭제: {project_name}")
                return True

        except Exception as e:
            logger.error(f"❌ 프로젝트 데이터 삭제 실패: {e}")
            return False

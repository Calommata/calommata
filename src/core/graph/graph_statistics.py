"""그래프 통계 모듈

그래프 통계 조회 및 프로젝트 관리를 담당합니다.
"""

import logging
from typing import Any

from src.graph import Neo4jPersistence

logger = logging.getLogger(__name__)


class GraphStatistics:
    """그래프 통계 조회 및 프로젝트 관리"""

    def __init__(self, persistence: Neo4jPersistence, project_name: str):
        self.persistence = persistence
        self.project_name = project_name

    def get_statistics(self) -> dict[str, Any]:
        """프로젝트 통계 정보 조회

        Returns:
            통계 정보 딕셔너리
        """
        try:
            stats = self.persistence.get_database_statistics()
            return stats
        except Exception as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return {}

    def clear_project(self) -> bool:
        """프로젝트 데이터 삭제

        Returns:
            성공 여부
        """
        try:
            result = self.persistence.clear_project_data(self.project_name)
            if result:
                logger.info(f"✅ 프로젝트 삭제 완료: {self.project_name}")
            return result
        except Exception as e:
            logger.error(f"❌ 프로젝트 삭제 실패: {e}")
            return False

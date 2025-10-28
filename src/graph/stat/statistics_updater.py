"""그래프 통계 업데이트 전문 클래스

그래프 통계 정보 계산 및 업데이트만 담당합니다.
"""

import logging

from src.graph.db import CodeGraph

logger = logging.getLogger(__name__)


class GraphStatisticsUpdater:
    """코드 그래프 통계 정보 업데이트 전문 클래스"""

    @staticmethod
    def update_statistics(graph: CodeGraph) -> None:
        """그래프 통계 정보 업데이트

        파일 수, 총 라인 수 등의 통계를 계산하여 그래프에 반영합니다.

        Args:
            graph: 업데이트할 CodeGraph
        """
        if not graph.nodes:
            return

        # 파일 수 계산
        file_paths = {node.file_path for node in graph.nodes.values()}
        graph.total_files = len(file_paths)

        # 총 라인 수 계산
        total_lines = sum(
            len(node.source_code.splitlines()) if node.source_code else 0
            for node in graph.nodes.values()
        )
        graph.total_lines = total_lines

        logger.debug(
            f"통계 업데이트: {graph.total_files}개 파일, {graph.total_lines}줄"
        )

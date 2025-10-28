"""프로젝트 분석 모듈

Parser와 Graph Adapter를 사용하여 프로젝트 또는 단일 파일을 분석합니다.
"""

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from src.parser import CodeASTAnalyzer
from src.graph import ParserToGraphAdapter, CodeGraph

logger = logging.getLogger(__name__)


class ProjectAnalyzer(BaseModel):
    """프로젝트 및 파일 분석"""

    project_name: str = Field(..., description="프로젝트 이름")

    class Config:
        arbitrary_types_allowed = True

    def analyze_and_store_project(self, project_path: str) -> CodeGraph:
        """프로젝트 분석 및 그래프 생성

        Args:
            project_path: 분석할 프로젝트 경로

        Returns:
            생성된 CodeGraph
        """
        logger.info(f"프로젝트 분석 시작: {project_path}")

        # 1. Parser로 코드 분석
        analyzer = CodeASTAnalyzer()
        code_blocks = analyzer.analyze_directory(project_path)
        logger.info(f"✅ {len(code_blocks)}개 코드 블록 추출 완료")

        # 2. Graph로 변환
        adapter = ParserToGraphAdapter()
        graph = adapter.convert_to_graph(
            code_blocks,
            project_name=self.project_name,
            project_path=project_path,
        )
        logger.info(f"✅ 그래프 변환 완료: {len(graph.nodes)}개 노드")

        return graph

    def analyze_and_store_file(self, file_path: str) -> CodeGraph:
        """단일 파일 분석 및 그래프 생성

        Args:
            file_path: 분석할 파일 경로

        Returns:
            생성된 CodeGraph
        """
        logger.info(f"파일 분석 시작: {file_path}")

        # 1. Parser로 코드 분석
        analyzer = CodeASTAnalyzer()
        code_blocks = analyzer.analyze_file(file_path)
        logger.info(f"✅ {len(code_blocks)}개 코드 블록 추출 완료")

        # 2. Graph로 변환
        adapter = ParserToGraphAdapter()
        graph = adapter.convert_to_graph(
            code_blocks,
            project_name=self.project_name,
            project_path=str(Path(file_path).parent),
        )
        logger.info(f"✅ 그래프 변환 완료: {len(graph.nodes)}개 노드")

        return graph

"""그래프 서비스 모듈

Parser와 Graph를 통합하여 코드 분석 및 임베딩을 관리합니다.
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.parser import CodeAnalyzer
from src.graph import ParserToGraphAdapter
from src.graph import Neo4jPersistence
from src.graph import CodeGraph

from .embedder import CodeEmbedder

logger = logging.getLogger(__name__)


class GraphService(BaseModel):
    """코드 그래프 분석 및 관리 서비스

    Parser로 코드 분석 -> Graph 변환 -> 임베딩 생성 -> Neo4j 저장
    전체 파이프라인을 관리합니다.
    """

    persistence: Any = Field(..., description="Neo4j 지속성 객체")
    embedder: CodeEmbedder = Field(..., description="코드 임베딩 생성기")
    project_name: str = Field(..., description="프로젝트 이름")

    class Config:
        arbitrary_types_allowed = True

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
        logger.info(f"프로젝트 분석 시작: {project_path}")

        # 1. Parser로 코드 분석
        analyzer = CodeAnalyzer()
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

        # 3. 임베딩 생성 (선택적)
        if create_embeddings:
            self._create_embeddings_for_graph(graph)

        # 4. 벡터 인덱스 최적화 (임베딩 차원에 맞게)
        if create_embeddings and graph.nodes:
            first_node = next(iter(graph.nodes.values()))
            if first_node.embedding_vector:
                embedding_dim = len(first_node.embedding_vector)
                self.persistence.create_vector_index_for_dimension(embedding_dim)
                logger.info(f"✅ {embedding_dim}차원 벡터 인덱스 최적화 완료")

        # 5. Neo4j에 저장
        self.persistence.save_code_graph(graph, self.project_name)
        logger.info(f"✅ Neo4j 저장 완료")

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
        logger.info(f"파일 분석 시작: {file_path}")

        # 1. Parser로 코드 분석
        analyzer = CodeAnalyzer()
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

        # 3. 임베딩 생성 (선택적)
        if create_embeddings:
            self._create_embeddings_for_graph(graph)

        # 4. Neo4j에 저장
        self.persistence.save_code_graph(graph, self.project_name)
        logger.info(f"✅ Neo4j 저장 완료")

        return graph

    def _create_embeddings_for_graph(self, graph: CodeGraph) -> None:
        """그래프의 모든 노드에 대해 임베딩 생성

        Args:
            graph: 임베딩을 생성할 CodeGraph
        """
        logger.info(f"임베딩 생성 시작: {len(graph.nodes)}개 노드")

        nodes = list(graph.nodes.values())

        # 배치로 처리
        batch_size = 32
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]

            # 코드 + docstring 결합하여 임베딩
            texts = []
            for node in batch:
                text = node.source_code
                if node.docstring:
                    text = f"{node.docstring}\n\n{node.source_code}"
                texts.append(text)

            # 임베딩 생성
            try:
                embeddings = self.embedder.embed_codes(texts)

                # 노드에 임베딩 저장
                for node, embedding in zip(batch, embeddings):
                    node.embedding_vector = embedding
                    node.embedding_model = self.embedder.model_name

                logger.info(
                    f"✅ 배치 {i // batch_size + 1} 임베딩 완료 ({len(batch)}개)"
                )

            except Exception as e:
                logger.error(f"❌ 배치 임베딩 실패: {e}")
                continue

        logger.info(f"✅ 전체 임베딩 생성 완료")

    def update_embeddings(
        self,
        node_ids: list[str] | None = None,
    ) -> None:
        """특정 노드들의 임베딩 업데이트

        Args:
            node_ids: 업데이트할 노드 ID 리스트 (None이면 모든 노드)
        """
        if node_ids is None:
            logger.warning("전체 노드 임베딩 업데이트는 아직 지원하지 않습니다")
            return

        logger.info(f"{len(node_ids)}개 노드 임베딩 업데이트 시작")

        for node_id in node_ids:
            try:
                # 노드 컨텍스트 조회
                context = self.persistence.get_node_context(node_id, depth=1)
                center_node = context.get("center_node")

                if not center_node:
                    logger.warning(f"노드를 찾을 수 없음: {node_id}")
                    continue

                # 임베딩 생성
                code = center_node.get("source_code", "")
                docstring = center_node.get("docstring", "")

                text = code
                if docstring:
                    text = f"{docstring}\n\n{code}"

                embedding = self.embedder.embed_code(text)

                # Neo4j 업데이트
                self.persistence.update_node_embedding(
                    node_id,
                    embedding,
                    self.embedder.model_name,
                )

                logger.info(f"✅ 노드 임베딩 업데이트: {node_id}")

            except Exception as e:
                logger.error(f"❌ 노드 {node_id} 임베딩 업데이트 실패: {e}")

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

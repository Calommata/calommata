"""그래프 임베딩 생성 모듈

그래프 노드의 임베딩을 생성하고 관리합니다.
"""

import logging


from src.graph import CodeGraph, Neo4jPersistence

from src.core.embedding.code_embedder import CodeEmbedder
from src.core.constants.embedding import DEFAULT_EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class GraphEmbedder:
    """그래프 노드 임베딩 생성 및 관리"""

    def __init__(self, embedder: CodeEmbedder, persistence: Neo4jPersistence):
        self.embedder = embedder
        self.persistence = persistence

    def create_embeddings_for_graph(self, graph: CodeGraph) -> None:
        """그래프의 모든 노드에 대해 임베딩 생성

        Args:
            graph: 임베딩을 생성할 CodeGraph
        """
        logger.info(f"임베딩 생성 시작: {len(graph.nodes)}개 노드")

        nodes = list(graph.nodes.values())

        # 배치로 처리
        for i in range(0, len(nodes), DEFAULT_EMBEDDING_BATCH_SIZE):
            batch = nodes[i : i + DEFAULT_EMBEDDING_BATCH_SIZE]

            # 코드 임베딩
            texts = []
            for node in batch:
                texts.append(node.source_code)

            # 임베딩 생성
            try:
                embeddings = self.embedder.embed_codes(texts)

                # 노드에 임베딩 저장
                for node, embedding in zip(batch, embeddings):
                    node.embedding_vector = embedding
                    node.embedding_model = self.embedder.model_name

                logger.info(
                    f"✅ 배치 {i // DEFAULT_EMBEDDING_BATCH_SIZE + 1} 임베딩 완료 ({len(batch)}개)"
                )

            except Exception as e:
                logger.error(f"❌ 배치 임베딩 실패: {e}")
                continue

        logger.info("✅ 전체 임베딩 생성 완료")

    def update_embeddings(self, node_ids: list[str] | None = None) -> None:
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
                embedding = self.embedder.embed_code(code)

                # Neo4j 업데이트
                self.persistence.update_node_embedding(
                    node_id,
                    embedding,
                    self.embedder.model_name,
                )

                logger.info(f"✅ 노드 임베딩 업데이트: {node_id}")

            except Exception as e:
                logger.error(f"❌ 노드 {node_id} 임베딩 업데이트 실패: {e}")

"""코드 검색 노드 모듈

임베딩을 사용하여 유사한 코드를 검색합니다.
"""

import logging
from typing import Any

from src.core.embedding import CodeEmbedder
from src.core.code_retriever import CodeRetriever
from src.core.state import AgentState

logger = logging.getLogger(__name__)


def code_retrieval_node(embedder: CodeEmbedder, retriever: CodeRetriever):
    def process(state: AgentState) -> dict[str, Any]:
        """검색 수행

        임베딩을 사용하여 유사한 코드를 검색합니다.

        Args:
            state: Agent 상태

        Returns:
            업데이트된 상태 딕셔너리
        """
        logger.info(f"코드 검색 단계: {state.query}")

        try:
            # 쿼리 임베딩 생성
            query_embedding = embedder.embed_code(state.query)

            # 유사한 코드 검색
            search_results = retriever.search_similar_code(
                query_embedding=query_embedding,
                limit=5,
            )
            logger.info(f"✅ {len(search_results)}개 코드 검색 완료")
            return {"search_results": search_results}

        except Exception as e:
            logger.error(f"❌ 코드 검색 실패: {e}")
            return {"search_results": []}

    return process

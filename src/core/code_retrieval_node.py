"""코드 검색 노드 모듈

임베딩을 사용하여 유사한 코드를 검색합니다.
"""

import logging
from typing import Any

from .embedder import CodeEmbedder
from .retriever import CodeRetriever
from .state import AgentState

logger = logging.getLogger(__name__)


class CodeRetrievalNode:
    """코드 검색 노드"""

    def __init__(self, embedder: CodeEmbedder, retriever: CodeRetriever):
        """초기화

        Args:
            embedder: 코드 임베딩 생성기
            retriever: 코드 검색 리트리버
        """
        self.embedder = embedder
        self.retriever = retriever

    def process(self, state: AgentState) -> dict[str, Any]:
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
            query_embedding = self.embedder.embed_code(state.query)

            # 유사한 코드 검색
            search_results = self.retriever.search_similar_code(
                query_embedding=query_embedding,
                limit=5,
            )
            logger.info(f"✅ {len(search_results)}개 코드 검색 완료")
            return {"search_results": search_results}

        except Exception as e:
            logger.error(f"❌ 코드 검색 실패: {e}")
            return {"search_results": []}

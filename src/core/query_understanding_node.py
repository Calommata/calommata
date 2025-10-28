"""쿼리 이해 노드 모듈

사용자 쿼리를 처리하고 임베딩을 생성합니다.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from .embedder import CodeEmbedder
from .state import AgentState

logger = logging.getLogger(__name__)


class QueryUnderstandingNode:
    """쿼리 이해 및 처리"""

    def __init__(self, embedder: CodeEmbedder):
        """초기화

        Args:
            embedder: 코드 임베딩 생성기
        """
        self.embedder = embedder

    def process(self, state: AgentState) -> dict[str, Any]:
        """쿼리 처리

        사용자 쿼리를 분석하고 임베딩을 생성합니다.

        Args:
            state: Agent 상태

        Returns:
            업데이트된 상태 딕셔너리
        """
        logger.info(f"쿼리 이해 단계: {state.query}")

        # 메시지에서 쿼리 추출 (첫 HumanMessage)
        query = state.query
        if not query and state.messages:
            for msg in state.messages:
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break

        return {"query": query}

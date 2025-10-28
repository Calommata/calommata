"""컨텍스트 구성 노드 모듈

검색된 코드들을 LLM이 이해하기 쉬운 형태로 구성합니다.
"""

import logging
from typing import Any

from .context_optimizer import ContextOptimizer
from .state import AgentState

logger = logging.getLogger(__name__)


class ContextBuildingNode:
    """컨텍스트 구성 노드"""

    def __init__(self, context_optimizer: ContextOptimizer | None = None):
        """초기화

        Args:
            context_optimizer: 컨텍스트 최적화기 (기본값: 새로 생성)
        """
        self.context_optimizer = context_optimizer or ContextOptimizer()

    def process(self, state: AgentState) -> dict[str, Any]:
        """컨텍스트 구성

        검색된 코드들을 LLM이 이해하기 쉬운 형태로 구성합니다.

        Args:
            state: Agent 상태

        Returns:
            업데이트된 상태 딕셔너리
        """
        logger.info("컨텍스트 구성 단계")

        if not state.search_results:
            return {"context": "No relevant code found."}

        # 최적화된 컨텍스트 생성
        context = self.context_optimizer.build_optimized_context(
            state.search_results,
            max_results=5,
        )

        logger.info(
            f"✅ 최적화된 컨텍스트 구성 완료 "
            f"(길이: {len(context)} 문자, {len(state.search_results)}개 중 최대 5개)"
        )
        return {"context": context}

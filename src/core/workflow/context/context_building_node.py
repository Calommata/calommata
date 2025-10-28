import logging
from typing import Any

from src.core.workflow.context.context_optimizer import ContextOptimizer
from src.core.state import AgentState

logger = logging.getLogger(__name__)


def context_building_node(state: AgentState) -> dict[str, Any]:
    """
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
    context_optimizer = ContextOptimizer()
    context = context_optimizer.build_optimized_context(
        state.search_results,
        max_results=5,
    )

    logger.info(
        f"✅ 최적화된 컨텍스트 구성 완료 "
        f"(길이: {len(context)} 문자, {len(state.search_results)}개 중 최대 5개)"
    )
    return {"context": context}

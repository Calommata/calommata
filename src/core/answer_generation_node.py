"""답변 생성 노드 모듈

LLM을 사용하여 최종 답변을 생성합니다.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .state import AgentState

logger = logging.getLogger(__name__)


class AnswerGenerationNode:
    """답변 생성 노드"""

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
    ):
        """초기화

        Args:
            llm: LangChain LLM 인스턴스
        """
        self.llm = llm

    def process(self, state: AgentState) -> dict[str, Any]:
        """답변 생성

        LLM을 사용하여 최종 답변을 생성합니다.

        Args:
            state: Agent 상태

        Returns:
            업데이트된 상태 딕셔너리
        """
        logger.info("답변 생성 단계")

        # 프롬프트 구성
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT_TEMPLATE),
            ]
        )

        try:
            # LLM 호출
            chain = prompt | self.llm
            response = chain.invoke(
                {
                    "query": state.query,
                    "context": state.context,
                }
            )

            final_answer = (
                response.content if hasattr(response, "content") else str(response)
            )

            logger.info(f"✅ 답변 생성 완료 (길이: {len(final_answer)})")

            # 메시지 추가
            new_messages = [AIMessage(content=final_answer)]

            return {
                "final_answer": final_answer,
                "messages": new_messages,
            }

        except Exception as e:
            logger.error(f"❌ 답변 생성 실패: {e}")
            error_message = f"답변 생성 중 오류가 발생했습니다: {e!s}"
            return {
                "final_answer": error_message,
                "messages": [AIMessage(content=error_message)],
            }

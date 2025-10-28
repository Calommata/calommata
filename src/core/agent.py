"""AI Agent 모듈

LangChain과 LangGraph를 사용하여 코드 검색 및 분석을 수행하는 AI Agent
"""

import logging
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from .embedder import CodeEmbedder
from .retriever import CodeRetriever, CodeSearchResult
from .state import AgentState
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .context_optimizer import ContextOptimizer

logger = logging.getLogger(__name__)


class CodeRAGAgent(BaseModel):
    """코드 검색 및 분석을 위한 RAG Agent

    1. 사용자 쿼리를 임베딩으로 변환
    2. Neo4j에서 유사한 코드 검색
    3. 그래프 관계를 탐색하여 연관 코드 수집
    4. LLM을 사용하여 답변 생성
    """

    embedder: CodeEmbedder = Field(..., description="코드 임베딩 생성기")
    retriever: CodeRetriever = Field(..., description="코드 검색 리트리버")
    llm_api_key: str = Field(..., description="Google Gemini API 키")
    model_name: str = Field(default="gemini-2.0-flash", description="사용할 LLM 모델")
    temperature: float = Field(default=0.7, description="LLM 온도")
    max_tokens: int = Field(default=4096, description="최대 토큰 수")

    _llm: Any = None
    _graph: Any = None
    _context_optimizer: ContextOptimizer | None = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        """Agent 초기화"""
        super().__init__(**data)
        self._context_optimizer = ContextOptimizer()
        self._initialize_llm()
        self._build_graph()

    def _initialize_llm(self) -> None:
        """LLM 모델 초기화"""
        try:
            logger.info(f"LLM 초기화 중: {self.model_name}")
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.llm_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            logger.info(f"✅ LLM 초기화 완료: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ LLM 초기화 실패: {e}")
            raise

    def _build_graph(self) -> None:
        """LangGraph 워크플로우 구성

        단계:
        1. query_understanding: 쿼리 이해 및 임베딩
        2. code_retrieval: 관련 코드 검색
        3. context_building: 컨텍스트 구성
        4. answer_generation: 답변 생성
        """
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("query_understanding", self._query_understanding_node)
        workflow.add_node("code_retrieval", self._code_retrieval_node)
        workflow.add_node("context_building", self._context_building_node)
        workflow.add_node("answer_generation", self._answer_generation_node)

        # 엣지 추가
        workflow.set_entry_point("query_understanding")
        workflow.add_edge("query_understanding", "code_retrieval")
        workflow.add_edge("code_retrieval", "context_building")
        workflow.add_edge("context_building", "answer_generation")
        workflow.add_edge("answer_generation", END)

        self._graph = workflow.compile()
        logger.info("✅ LangGraph 워크플로우 구성 완료")

    def _query_understanding_node(self, state: AgentState) -> dict[str, Any]:
        """쿼리 이해 노드

        사용자 쿼리를 분석하고 임베딩을 생성합니다.
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

    def _code_retrieval_node(self, state: AgentState) -> dict[str, Any]:
        """코드 검색 노드

        임베딩을 사용하여 유사한 코드를 검색합니다.
        """
        logger.info(f"코드 검색 단계: {state.query}")

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedder.embed_code(state.query)

            # 유사한 코드 검색 (순수 벡터 검색)
            search_results = self.retriever.search_similar_code(
                query_embedding=query_embedding,
                limit=5,
            )
            logger.info(f"✅ {len(search_results)}개 코드 검색 완료")
            return {"search_results": search_results}

        except Exception as e:
            logger.error(f"❌ 코드 검색 실패: {e}")
            return {"search_results": []}

    def _context_building_node(self, state: AgentState) -> dict[str, Any]:
        """컨텍스트 구성 노드

        검색된 코드들을 LLM이 이해하기 쉬운 형태로 구성합니다.
        """
        logger.info("컨텍스트 구성 단계")

        if not state.search_results:
            return {"context": "No relevant code found."}

        # 컨텍스트 최적화기를 사용하여 압축된 컨텍스트 생성
        if self._context_optimizer:
            context = self._context_optimizer.build_optimized_context(
                state.search_results,
                max_results=5,  # 최대 5개 결과만 포함
            )
        else:
            # Fallback: simple context building
            context = "\n\n".join(
                f"## {result.file_path}\n{result.source_code}"
                for result in state.search_results[:5]
            )

        logger.info(
            f"✅ 최적화된 컨텍스트 구성 완료 "
            f"(길이: {len(context)} 문자, {len(state.search_results)}개 중 최대 5개)"
        )
        return {"context": context}

    def _answer_generation_node(self, state: AgentState) -> dict[str, Any]:
        """답변 생성 노드

        LLM을 사용하여 최종 답변을 생성합니다.
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
            chain = prompt | self._llm
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

    # TODO : 추후 비동기처리도 고려하면 좋을 것 같지만, MVP 단계에서는 동기처리로 충분할 듯 합니다.
    def query(
        self,
        user_query: str,
        conversation_history: list[BaseMessage] | None = None,
    ) -> str:
        """사용자 쿼리 처리

        Args:
            user_query: 사용자 질문
            conversation_history: 이전 대화 내역 (선택적)

        Returns:
            AI Agent의 답변
        """
        logger.info(f"쿼리 처리 시작: {user_query}")

        # 초기 상태 구성
        messages = conversation_history or []
        messages.append(HumanMessage(content=user_query))

        initial_state = AgentState(
            messages=messages,
            query=user_query,
        )

        try:
            # 그래프 실행
            final_state = self._graph.invoke(initial_state)

            answer = final_state.get("final_answer", "답변을 생성할 수 없습니다.")
            logger.info("✅ 쿼리 처리 완료")

            return answer

        except Exception as e:
            logger.error(f"❌ 쿼리 처리 실패: {e!r}")
            return f"쿼리 처리 중 오류가 발생했습니다: {e!s}"

    def get_search_results(self, user_query: str) -> list[CodeSearchResult]:
        """검색 결과만 반환 (답변 생성 없이)

        Args:
            user_query: 사용자 질문

        Returns:
            검색 결과 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedder.embed_code(user_query)

            # 유사한 코드 검색
            search_results = self.retriever.search_similar_code(
                query_embedding=query_embedding,
            )

            logger.info(f"✅ {len(search_results)}개 검색 결과 반환")
            return search_results

        except Exception as e:
            logger.error(f"❌ 검색 실패: {e!r}")
            return []

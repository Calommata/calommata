import logging
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.core.embedding.code_embedder import CodeEmbedder
from src.core.code_retriever import CodeRetriever, CodeSearchResult
from src.core.state import AgentState
from src.core.workflow import (
    code_retrieval_node,
    context_building_node,
    answer_generation_node,
)

logger = logging.getLogger(__name__)


class CodeRAGAgent:
    """코드 검색 및 분석을 위한 RAG Agent 오케스트레이터

    3개의 노드를 LangGraph로 연결하여 검색 및 답변 생성을 수행합니다:
    1. 코드 검색
    2. 컨텍스트 구성
    3. 답변 생성
    """

    def __init__(
        self,
        embedder: CodeEmbedder,
        retriever: CodeRetriever,
        llm_api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Agent 초기화

        Args:
            embedder: 코드 임베딩 생성기
            retriever: 코드 검색 리트리버
            llm_api_key: Google Gemini API 키
            model_name: 사용할 LLM 모델
            temperature: LLM 온도
            max_tokens: 최대 토큰 수
        """
        self.embedder = embedder
        self.retriever = retriever
        self.llm_api_key = llm_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._initialize_llm()
        self._initialize_nodes()
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

    def _initialize_nodes(self) -> None:
        """노드 인스턴스 초기화"""
        if not self._llm:
            raise RuntimeError("LLM이 초기화되지 않았습니다")

        self._retrieval_node = code_retrieval_node(self.embedder, self.retriever)
        self._context_node = context_building_node
        self._answer_node = answer_generation_node(self._llm)
        logger.info("✅ 모든 노드 초기화 완료")

    def _build_graph(self) -> None:
        """LangGraph 워크플로우 구성

        3개 노드를 순차적으로 연결합니다:
        code_retrieval -> context_building -> answer_generation
        """
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("code_retrieval", self._process_retrieval_node)
        workflow.add_node("context_building", self._process_context_node)
        workflow.add_node("answer_generation", self._process_answer_node)

        # 엣지 추가 (선형 파이프라인)
        workflow.set_entry_point("code_retrieval")
        workflow.add_edge("code_retrieval", "context_building")
        workflow.add_edge("context_building", "answer_generation")
        workflow.add_edge("answer_generation", END)

        self._graph = workflow.compile()
        logger.info("✅ LangGraph 워크플로우 구성 완료")

    def _process_retrieval_node(self, state: AgentState) -> dict[str, Any]:
        """검색 처리 노드"""
        return self._retrieval_node(state)

    def _process_context_node(self, state: AgentState) -> dict[str, Any]:
        """컨텍스트 처리 노드"""
        return self._context_node(state)

    def _process_answer_node(self, state: AgentState) -> dict[str, Any]:
        """답변 처리 노드"""
        return self._answer_node(state)

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
        messages = list(conversation_history) if conversation_history else []
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

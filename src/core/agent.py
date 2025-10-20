"""AI Agent 모듈

LangChain과 LangGraph를 사용하여 코드 검색 및 분석을 수행하는 AI Agent
"""

import logging
from typing import Any, Annotated, Sequence

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .embedder import CodeEmbedder
from .retriever import CodeRetriever, CodeSearchResult

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """Agent 상태 관리

    LangGraph에서 사용하는 상태 객체
    """

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list, description="대화 메시지 리스트"
    )

    query: str = Field(default="", description="사용자 쿼리")
    search_results: list[CodeSearchResult] = Field(
        default_factory=list, description="검색 결과"
    )
    context: str = Field(default="", description="생성된 컨텍스트")
    final_answer: str = Field(default="", description="최종 답변")

    class Config:
        arbitrary_types_allowed = True


class CodeRAGAgent(BaseModel):
    """코드 검색 및 분석을 위한 RAG Agent

    LangChain과 LangGraph를 사용하여 다음을 수행합니다:
    1. 사용자 쿼리를 임베딩으로 변환
    2. Neo4j에서 유사한 코드 검색
    3. 그래프 관계를 탐색하여 연관 코드 수집
    4. LLM을 사용하여 답변 생성
    """

    embedder: CodeEmbedder = Field(..., description="코드 임베딩 생성기")
    retriever: CodeRetriever = Field(..., description="코드 검색 리트리버")
    llm_api_key: str = Field(..., description="Google Gemini API 키")
    model_name: str = Field(default="gemini-2.5-flash", description="사용할 LLM 모델")
    temperature: float = Field(default=0.1, description="LLM 온도")
    max_tokens: int = Field(default=4096, description="최대 토큰 수")

    _llm: Any = None
    _graph: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data: Any):
        """Agent 초기화"""
        super().__init__(**data)
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

            # 유사한 코드 검색 (GraphRAG 강화)
            search_results = self.retriever.search_similar_code(
                query_embedding=query_embedding,
                limit=5,
                expand_results=True,  # 그래프 확장 검색 활성화
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
            return {"context": "관련된 코드를 찾을 수 없습니다."}

        # 검색 결과를 문자열로 변환
        context_parts = ["# 검색된 관련 코드\n"]

        for i, result in enumerate(state.search_results, 1):
            context_parts.append(f"\n## 결과 {i}\n")
            context_parts.append(result.to_context_string())

        context = "\n".join(context_parts)

        logger.info(f"✅ 컨텍스트 구성 완료 (길이: {len(context)})")
        return {"context": context}

    def _answer_generation_node(self, state: AgentState) -> dict[str, Any]:
        """답변 생성 노드

        LLM을 사용하여 최종 답변을 생성합니다.
        """
        logger.info("답변 생성 단계")

        # 프롬프트 구성
        system_prompt = """당신은 코드 분석 전문가입니다.
사용자의 질문에 대해 제공된 코드 컨텍스트를 분석하여 정확하고 유용한 답변을 제공하세요.

답변 시 다음을 포함하세요:
1. 질문에 대한 직접적인 답변
2. 관련 코드 설명
3. 코드 간의 관계 및 의존성
4. 필요한 경우 사용 예시

코드 블록을 사용할 때는 적절한 언어를 명시하세요 (```python)."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """**사용자 질문:**
{query}

**검색된 코드 컨텍스트:**
{context}

위 정보를 바탕으로 사용자의 질문에 답변해주세요.""",
                ),
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
            logger.info(f"✅ 쿼리 처리 완료")

            return answer

        except Exception as e:
            logger.error(f"❌ 쿼리 처리 실패: {e!r}")
            return f"쿼리 처리 중 오류가 발생했습니다: {e!s}"

    async def aquery(
        self,
        user_query: str,
        conversation_history: list[BaseMessage] | None = None,
    ) -> str:
        """비동기 쿼리 처리

        Args:
            user_query: 사용자 질문
            conversation_history: 이전 대화 내역 (선택적)

        Returns:
            AI Agent의 답변
        """
        logger.info(f"비동기 쿼리 처리 시작: {user_query}")

        # 초기 상태 구성
        messages = conversation_history or []
        messages.append(HumanMessage(content=user_query))

        initial_state = AgentState(
            messages=messages,
            query=user_query,
        )

        try:
            # 비동기 그래프 실행
            final_state = await self._graph.ainvoke(initial_state)

            answer = final_state.get("final_answer", "답변을 생성할 수 없습니다.")
            logger.info(f"✅ 비동기 쿼리 처리 완료")

            return answer

        except Exception as e:
            logger.error(f"❌ 비동기 쿼리 처리 실패: {e!r}")
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

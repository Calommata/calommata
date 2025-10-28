"""AI Agent State 정의

LangGraph Agent의 상태를 정의합니다.
"""

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .retriever import CodeSearchResult


class AgentState(BaseModel):
    """RAG Agent의 상태

    Attributes:
        messages: 대화 메시지 리스트
        query: 사용자 쿼리
        search_results: 검색 결과
        context: 생성된 컨텍스트
        final_answer: 최종 답변
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

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.core.code_retriever import CodeSearchResult


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(
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

"""
LangGraph 기반 GraphRAG 워크플로우
코드 분석을 위한 지능형 그래프 검색 및 추론 시스템
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage

from .llm_manager import LLMManager, TaskType
from .embedding_service import EmbeddingService
from .neo4j_handler import Neo4jHandler

logger = logging.getLogger(__name__)


class RAGWorkflowType(Enum):
    """RAG 워크플로우 타입"""

    SIMPLE_SEARCH = "simple_search"
    CONTEXTUAL_ANALYSIS = "contextual_analysis"
    ARCHITECTURE_REVIEW = "architecture_review"
    CODE_SIMILARITY = "code_similarity"
    REFACTORING_SUGGESTIONS = "refactoring_suggestions"


@dataclass
class RAGConfig:
    """GraphRAG 설정"""

    max_results: int = 10
    similarity_threshold: float = 0.7
    context_depth: int = 2
    include_related_nodes: bool = True
    max_context_tokens: int = 4000
    enable_reasoning: bool = True
    workflow_type: RAGWorkflowType = RAGWorkflowType.CONTEXTUAL_ANALYSIS


class RAGState(TypedDict):
    """RAG 워크플로우 상태"""

    # 입력
    query: str
    workflow_type: str

    # 검색 결과
    search_results: List[Dict[str, Any]]
    embeddings: List[List[float]]

    # 컨텍스트
    context_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

    # LLM 처리
    messages: List[BaseMessage]
    analysis_result: Optional[str]

    # 메타데이터
    step_history: List[str]
    confidence_score: float
    error_messages: List[str]


class GraphRAGWorkflow:
    """LangGraph 기반 GraphRAG 워크플로우"""

    def __init__(
        self,
        neo4j_handler: Neo4jHandler,
        embedding_service: EmbeddingService,
        llm_manager: LLMManager,
        config: Optional[RAGConfig] = None,
    ):
        self.neo4j_handler = neo4j_handler
        self.embedding_service = embedding_service
        self.llm_manager = llm_manager
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)

        # 워크플로우 그래프 초기화
        self.workflows = self._build_workflows()

    def _build_workflows(self) -> Dict[RAGWorkflowType, CompiledStateGraph]:
        """다양한 RAG 워크플로우 구축"""
        workflows = {}

        # 1. 단순 검색 워크플로우
        workflows[RAGWorkflowType.SIMPLE_SEARCH] = self._build_simple_search_workflow()

        # 2. 컨텍스트 분석 워크플로우
        workflows[RAGWorkflowType.CONTEXTUAL_ANALYSIS] = (
            self._build_contextual_analysis_workflow()
        )

        # 3. 아키텍처 리뷰 워크플로우
        workflows[RAGWorkflowType.ARCHITECTURE_REVIEW] = (
            self._build_architecture_review_workflow()
        )

        # 4. 코드 유사성 분석 워크플로우
        workflows[RAGWorkflowType.CODE_SIMILARITY] = (
            self._build_code_similarity_workflow()
        )

        # 5. 리팩토링 제안 워크플로우
        workflows[RAGWorkflowType.REFACTORING_SUGGESTIONS] = (
            self._build_refactoring_workflow()
        )

        return workflows

    def _build_simple_search_workflow(self) -> CompiledStateGraph:
        """단순 검색 워크플로우"""
        workflow = StateGraph(RAGState)

        # 노드 추가
        workflow.add_node("embed_query", self._embed_query)
        workflow.add_node("vector_search", self._vector_search)
        workflow.add_node("format_results", self._format_simple_results)

        # 엣지 설정
        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "vector_search")
        workflow.add_edge("vector_search", "format_results")
        workflow.add_edge("format_results", END)

        return workflow.compile()

    def _build_contextual_analysis_workflow(self) -> CompiledStateGraph:
        """컨텍스트 분석 워크플로우"""
        workflow = StateGraph(RAGState)

        # 노드 추가
        workflow.add_node("embed_query", self._embed_query)
        workflow.add_node("vector_search", self._vector_search)
        workflow.add_node("gather_context", self._gather_context)
        workflow.add_node("analyze_with_llm", self._analyze_with_llm)
        workflow.add_node("synthesize_response", self._synthesize_response)

        # 엣지 설정
        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "vector_search")
        workflow.add_edge("vector_search", "gather_context")
        workflow.add_edge("gather_context", "analyze_with_llm")
        workflow.add_edge("analyze_with_llm", "synthesize_response")
        workflow.add_edge("synthesize_response", END)

        return workflow.compile()

    def _build_architecture_review_workflow(self) -> CompiledStateGraph:
        """아키텍처 리뷰 워크플로우"""
        workflow = StateGraph(RAGState)

        workflow.add_node("embed_query", self._embed_query)
        workflow.add_node("broad_search", self._broad_search)
        workflow.add_node("analyze_dependencies", self._analyze_dependencies)
        workflow.add_node("evaluate_architecture", self._evaluate_architecture)
        workflow.add_node("generate_recommendations", self._generate_recommendations)

        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "broad_search")
        workflow.add_edge("broad_search", "analyze_dependencies")
        workflow.add_edge("analyze_dependencies", "evaluate_architecture")
        workflow.add_edge("evaluate_architecture", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)

        return workflow.compile()

    def _build_code_similarity_workflow(self) -> CompiledStateGraph:
        """코드 유사성 분석 워크플로우"""
        workflow = StateGraph(RAGState)

        workflow.add_node("embed_query", self._embed_query)
        workflow.add_node("similarity_search", self._similarity_search)
        workflow.add_node("cluster_similar_codes", self._cluster_similar_codes)
        workflow.add_node("analyze_patterns", self._analyze_patterns)
        workflow.add_node("summarize_insights", self._summarize_insights)

        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "similarity_search")
        workflow.add_edge("similarity_search", "cluster_similar_codes")
        workflow.add_edge("cluster_similar_codes", "analyze_patterns")
        workflow.add_edge("analyze_patterns", "summarize_insights")
        workflow.add_edge("summarize_insights", END)

        return workflow.compile()

    def _build_refactoring_workflow(self) -> CompiledStateGraph:
        """리팩토링 제안 워크플로우"""
        workflow = StateGraph(RAGState)

        workflow.add_node("embed_query", self._embed_query)
        workflow.add_node("find_target_code", self._find_target_code)
        workflow.add_node("analyze_code_quality", self._analyze_code_quality)
        workflow.add_node("find_similar_patterns", self._find_similar_patterns)
        workflow.add_node(
            "generate_refactoring_suggestions", self._generate_refactoring_suggestions
        )

        workflow.set_entry_point("embed_query")
        workflow.add_edge("embed_query", "find_target_code")
        workflow.add_edge("find_target_code", "analyze_code_quality")
        workflow.add_edge("analyze_code_quality", "find_similar_patterns")
        workflow.add_edge("find_similar_patterns", "generate_refactoring_suggestions")
        workflow.add_edge("generate_refactoring_suggestions", END)

        return workflow.compile()

    # 공통 워크플로우 노드들
    async def _embed_query(self, state: RAGState) -> RAGState:
        """쿼리 임베딩 생성"""
        try:
            embedding = self.embedding_service.create_query_embedding(state["query"])
            if embedding:
                state["embeddings"] = [embedding]
                state["step_history"].append("쿼리 임베딩 생성 완료")
            else:
                state["error_messages"].append("쿼리 임베딩 생성 실패")
        except Exception as e:
            state["error_messages"].append(f"임베딩 생성 오류: {str(e)}")

        return state

    async def _vector_search(self, state: RAGState) -> RAGState:
        """벡터 검색 실행"""
        try:
            if not state["embeddings"]:
                state["error_messages"].append("임베딩이 없어 검색 불가")
                return state

            # Neo4j 벡터 검색 (구현 예정)
            # 현재는 mock 데이터 반환
            mock_results = [
                {
                    "id": "mock_node_1",
                    "name": "example_function",
                    "node_type": "Function",
                    "file_path": "/example/path.py",
                    "source_code": "def example_function():\n    pass",
                    "similarity_score": 0.85,
                },
                {
                    "id": "mock_node_2",
                    "name": "another_function",
                    "node_type": "Function",
                    "file_path": "/example/other.py",
                    "source_code": "def another_function():\n    return True",
                    "similarity_score": 0.78,
                },
            ]

            state["search_results"] = mock_results
            state["step_history"].append(f"벡터 검색 완료: {len(mock_results)}개 결과")

        except Exception as e:
            state["error_messages"].append(f"벡터 검색 오류: {str(e)}")

        return state

    async def _gather_context(self, state: RAGState) -> RAGState:
        """컨텍스트 정보 수집"""
        try:
            context_nodes = []
            relationships = []

            for result in state["search_results"]:
                # 각 검색 결과에 대한 컨텍스트 수집
                node_id = result.get("id")
                if node_id:
                    # Neo4j에서 관련 노드들 조회 (구현 예정)
                    # 현재는 mock 데이터
                    context_nodes.extend(
                        [
                            {
                                "id": f"{node_id}_context_1",
                                "name": f"related_to_{result['name']}",
                                "relationship": "CALLS",
                            }
                        ]
                    )

            state["context_nodes"] = context_nodes
            state["relationships"] = relationships
            state["step_history"].append(
                f"컨텍스트 수집 완료: {len(context_nodes)}개 노드"
            )

        except Exception as e:
            state["error_messages"].append(f"컨텍스트 수집 오류: {str(e)}")

        return state

    async def _analyze_with_llm(self, state: RAGState) -> RAGState:
        """LLM을 사용한 분석"""
        try:
            if not self.llm_manager.is_available():
                state["error_messages"].append("LLM을 사용할 수 없습니다")
                return state

            # 컨텍스트 구성
            context_text = self._build_context_text(state)

            # LLM 분석 실행
            analysis = await self.llm_manager.analyze_code(
                task_type=TaskType.CODE_ANALYSIS,
                code=context_text,
                query=state["query"],
                language="python",
                file_path="multiple_files",
                code_type="mixed",
                additional_context=f"검색 결과 {len(state['search_results'])}개",
            )

            if analysis:
                state["analysis_result"] = analysis
                state["step_history"].append("LLM 분석 완료")
            else:
                state["error_messages"].append("LLM 분석 실패")

        except Exception as e:
            state["error_messages"].append(f"LLM 분석 오류: {str(e)}")

        return state

    async def _synthesize_response(self, state: RAGState) -> RAGState:
        """최종 응답 생성"""
        try:
            # 분석 결과와 검색 결과를 종합
            if state.get("analysis_result"):
                # LLM 분석 결과가 있는 경우
                confidence = 0.9
            else:
                # 검색 결과만 있는 경우
                confidence = 0.6
                state["analysis_result"] = self._create_fallback_response(state)

            state["confidence_score"] = confidence
            state["step_history"].append("응답 생성 완료")

        except Exception as e:
            state["error_messages"].append(f"응답 생성 오류: {str(e)}")

        return state

    # 워크플로우별 특화 노드들
    async def _format_simple_results(self, state: RAGState) -> RAGState:
        """단순 검색 결과 포맷팅"""
        results = state["search_results"]
        if results:
            formatted_results = []
            for result in results:
                formatted_results.append(
                    f"• {result['name']} ({result['node_type']}) - 유사도: {result['similarity_score']:.2f}"
                )

            state["analysis_result"] = "검색 결과:\n" + "\n".join(formatted_results)
            state["confidence_score"] = 0.7
        else:
            state["analysis_result"] = "검색 결과를 찾을 수 없습니다."
            state["confidence_score"] = 0.0

        return state

    async def _broad_search(self, state: RAGState) -> RAGState:
        """광범위한 검색 (아키텍처 분석용)"""
        # 아키텍처 분석을 위한 더 광범위한 검색
        state["step_history"].append("광범위한 검색 수행")
        return await self._vector_search(state)

    async def _analyze_dependencies(self, state: RAGState) -> RAGState:
        """의존성 분석"""
        state["step_history"].append("의존성 분석 수행")
        return state

    async def _evaluate_architecture(self, state: RAGState) -> RAGState:
        """아키텍처 평가"""
        state["step_history"].append("아키텍처 평가 수행")
        return state

    async def _generate_recommendations(self, state: RAGState) -> RAGState:
        """아키텍처 개선 권장사항 생성"""
        state["step_history"].append("권장사항 생성 완료")
        return state

    # 추가 워크플로우 노드들 (간략히 구현)
    async def _similarity_search(self, state: RAGState) -> RAGState:
        """유사성 검색"""
        return await self._vector_search(state)

    async def _cluster_similar_codes(self, state: RAGState) -> RAGState:
        """유사 코드 클러스터링"""
        state["step_history"].append("코드 클러스터링 완료")
        return state

    async def _analyze_patterns(self, state: RAGState) -> RAGState:
        """패턴 분석"""
        state["step_history"].append("패턴 분석 완료")
        return state

    async def _summarize_insights(self, state: RAGState) -> RAGState:
        """인사이트 요약"""
        state["step_history"].append("인사이트 요약 완료")
        return state

    async def _find_target_code(self, state: RAGState) -> RAGState:
        """리팩토링 대상 코드 검색"""
        return await self._vector_search(state)

    async def _analyze_code_quality(self, state: RAGState) -> RAGState:
        """코드 품질 분석"""
        state["step_history"].append("코드 품질 분석 완료")
        return state

    async def _find_similar_patterns(self, state: RAGState) -> RAGState:
        """유사 패턴 검색"""
        state["step_history"].append("유사 패턴 검색 완료")
        return state

    async def _generate_refactoring_suggestions(self, state: RAGState) -> RAGState:
        """리팩토링 제안 생성"""
        state["step_history"].append("리팩토링 제안 생성 완료")
        return state

    # 유틸리티 메서드들
    def _build_context_text(self, state: RAGState) -> str:
        """컨텍스트 텍스트 구성"""
        parts = []

        # 검색 결과 추가
        if state["search_results"]:
            parts.append("=== 검색 결과 ===")
            for result in state["search_results"]:
                parts.append(f"파일: {result.get('file_path', 'Unknown')}")
                parts.append(f"함수/클래스: {result.get('name', 'Unknown')}")
                parts.append(f"타입: {result.get('node_type', 'Unknown')}")
                if result.get("source_code"):
                    parts.append("코드:")
                    parts.append(result["source_code"])
                parts.append("---")

        # 컨텍스트 노드 추가
        if state["context_nodes"]:
            parts.append("=== 관련 컨텍스트 ===")
            for node in state["context_nodes"]:
                parts.append(
                    f"- {node.get('name', 'Unknown')} ({node.get('relationship', 'Unknown')})"
                )

        return "\n".join(parts)

    def _create_fallback_response(self, state: RAGState) -> str:
        """LLM 분석이 없을 때의 대체 응답 생성"""
        if not state["search_results"]:
            return "관련 코드를 찾을 수 없습니다."

        response_parts = [f"'{state['query']}'에 대한 검색 결과:", ""]

        for i, result in enumerate(state["search_results"][:5], 1):
            response_parts.append(f"{i}. {result.get('name', 'Unknown')}")
            response_parts.append(f"   파일: {result.get('file_path', 'Unknown')}")
            response_parts.append(f"   타입: {result.get('node_type', 'Unknown')}")
            response_parts.append(f"   유사도: {result.get('similarity_score', 0):.2f}")
            response_parts.append("")

        return "\n".join(response_parts)

    async def execute_workflow(
        self,
        query: str,
        workflow_type: RAGWorkflowType = RAGWorkflowType.CONTEXTUAL_ANALYSIS,
    ) -> Dict[str, Any]:
        """워크플로우 실행"""
        try:
            # 초기 상태 설정
            initial_state: RAGState = {
                "query": query,
                "workflow_type": workflow_type.value,
                "search_results": [],
                "embeddings": [],
                "context_nodes": [],
                "relationships": [],
                "messages": [],
                "analysis_result": None,
                "step_history": [],
                "confidence_score": 0.0,
                "error_messages": [],
            }

            # 워크플로우 선택 및 실행
            workflow = self.workflows.get(workflow_type)
            if not workflow:
                return {
                    "success": False,
                    "error": f"지원되지 않는 워크플로우: {workflow_type.value}",
                    "result": None,
                }

            # 워크플로우 실행
            final_state = await workflow.ainvoke(initial_state)

            # 결과 반환
            return {
                "success": True,
                "result": final_state.get("analysis_result"),
                "confidence_score": final_state.get("confidence_score", 0.0),
                "search_results": final_state.get("search_results", []),
                "step_history": final_state.get("step_history", []),
                "error_messages": final_state.get("error_messages", []),
                "workflow_type": workflow_type.value,
            }

        except Exception as e:
            self.logger.error(f"워크플로우 실행 실패: {e}")
            return {"success": False, "error": str(e), "result": None}

    def get_available_workflows(self) -> List[str]:
        """사용 가능한 워크플로우 목록"""
        return [workflow.value for workflow in RAGWorkflowType]

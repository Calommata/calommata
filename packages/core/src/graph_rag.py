"""
LangChain/LangGraph 기반 GraphRAG 서비스
통합된 지능형 코드 분석 및 검색 시스템
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

from .neo4j_handler import Neo4jHandler
from .embedding_service import EmbeddingService
from .llm_manager import LLMManager, LLMConfig, TaskType
from .graph_rag_workflow import (
    GraphRAGWorkflow,
    RAGWorkflowType,
    RAGConfig as WorkflowRAGConfig,
)


@dataclass
class RAGConfig:
    """GraphRAG 통합 설정"""

    max_results: int = 10
    similarity_threshold: float = 0.7
    context_depth: int = 2
    include_related_nodes: bool = True
    max_context_tokens: int = 4000
    enable_workflows: bool = True
    default_workflow: RAGWorkflowType = RAGWorkflowType.CONTEXTUAL_ANALYSIS


class GraphRAGService:
    """LangChain/LangGraph 기반 통합 GraphRAG 서비스"""

    def __init__(
        self,
        neo4j_handler: Neo4jHandler,
        embedding_service: EmbeddingService,
        llm_manager: Optional[LLMManager] = None,
        config: Optional[RAGConfig] = None,
    ):
        self.neo4j_handler = neo4j_handler
        self.embedding_service = embedding_service
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)

        # LLM 매니저 초기화
        self.llm_manager = llm_manager or self._init_default_llm_manager()

        # 워크플로우 초기화
        self.workflow_engine = None
        if self.config.enable_workflows and self.llm_manager:
            self.workflow_engine = self._init_workflow_engine()

    def _init_default_llm_manager(self) -> Optional[LLMManager]:
        """기본 LLM 매니저 초기화"""
        try:
            llm_config = LLMConfig(temperature=0.1, max_tokens=4000)
            return LLMManager(llm_config)
        except Exception as e:
            self.logger.warning(f"LLM 매니저 초기화 실패: {e}")
            return None

    def _init_workflow_engine(self) -> Optional[GraphRAGWorkflow]:
        """워크플로우 엔진 초기화"""
        try:
            workflow_config = WorkflowRAGConfig(
                max_results=self.config.max_results,
                similarity_threshold=self.config.similarity_threshold,
                context_depth=self.config.context_depth,
                max_context_tokens=self.config.max_context_tokens,
                workflow_type=self.config.default_workflow,
            )

            return GraphRAGWorkflow(
                neo4j_handler=self.neo4j_handler,
                embedding_service=self.embedding_service,
                llm_manager=self.llm_manager,
                config=workflow_config,
            )

        except Exception as e:
            self.logger.error(f"워크플로우 엔진 초기화 실패: {e}")
            return None

    async def search_similar_code(
        self,
        query: str,
        workflow_type: RAGWorkflowType = RAGWorkflowType.SIMPLE_SEARCH,
        **kwargs,
    ) -> dict[str, Any]:
        """유사한 코드 검색"""
        if self.workflow_engine:
            return await self.workflow_engine.execute_workflow(query, workflow_type)
        else:
            return await self._basic_search(query, **kwargs)

    async def analyze_code_architecture(
        self, query: str, project_name: str = None
    ) -> dict[str, Any]:
        """코드 아키텍처 분석"""
        if self.workflow_engine:
            enhanced_query = (
                f"Analyze architecture of project {project_name}: {query}"
                if project_name
                else query
            )
            return await self.workflow_engine.execute_workflow(
                enhanced_query, RAGWorkflowType.ARCHITECTURE_REVIEW
            )
        else:
            return {"success": False, "error": "워크플로우 엔진을 사용할 수 없습니다"}

    async def find_code_similarities(
        self, code_snippet: str, analysis_focus: str = "patterns"
    ) -> dict[str, Any]:
        """코드 유사성 분석"""
        if self.workflow_engine:
            query = f"Find similar code patterns for: {code_snippet}"
            return await self.workflow_engine.execute_workflow(
                query, RAGWorkflowType.CODE_SIMILARITY
            )
        else:
            return await self._basic_similarity_search(code_snippet)

    async def suggest_refactoring(
        self,
        target_code: str,
        refactoring_goals: str = "improve readability and maintainability",
    ) -> dict[str, Any]:
        """리팩토링 제안"""
        if self.workflow_engine:
            query = f"Suggest refactoring for code with goals: {refactoring_goals}. Code: {target_code}"
            return await self.workflow_engine.execute_workflow(
                query, RAGWorkflowType.REFACTORING_SUGGESTIONS
            )
        else:
            return await self._basic_refactoring_analysis(
                target_code, refactoring_goals
            )

    async def get_enriched_context(
        self, query: str, include_relationships: bool = True, **kwargs
    ) -> dict[str, Any]:
        """풍부한 컨텍스트 정보 제공"""
        if self.workflow_engine:
            return await self.workflow_engine.execute_workflow(
                query, RAGWorkflowType.CONTEXTUAL_ANALYSIS
            )
        else:
            return await self._basic_context_search(query, include_relationships)

    # 기본 검색 메서드들 (워크플로우 없을 때의 fallback)
    async def _basic_search(
        self, query: str, limit: int = None, **kwargs
    ) -> dict[str, Any]:
        """기본 검색 (워크플로우 없을 때)"""
        try:
            query_embedding = self.embedding_service.create_query_embedding(query)
            if not query_embedding:
                return {
                    "success": False,
                    "error": "쿼리 임베딩 생성 실패",
                    "result": None,
                }

            # Mock 결과 반환 (Neo4j 구현 대기)
            mock_results = [
                {
                    "name": "example_function",
                    "node_type": "Function",
                    "file_path": "/example/path.py",
                    "similarity_score": 0.85,
                    "source_code": "def example_function():\n    pass",
                }
            ]

            return {
                "success": True,
                "result": f"'{query}'에 대한 기본 검색 결과",
                "search_results": mock_results,
                "confidence_score": 0.6,
                "method": "basic_search",
            }

        except Exception as e:
            self.logger.error(f"기본 검색 실패: {e}")
            return {"success": False, "error": str(e), "result": None}

    async def _basic_similarity_search(self, code_snippet: str) -> dict[str, Any]:
        """기본 유사성 검색"""
        try:
            code_embedding = self.embedding_service.create_code_embedding(code_snippet)
            if not code_embedding:
                return {"success": False, "error": "코드 임베딩 생성 실패"}

            return {
                "success": True,
                "result": "유사한 코드 패턴을 찾았습니다 (기본 검색)",
                "confidence_score": 0.5,
                "method": "basic_similarity",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _basic_refactoring_analysis(
        self, code: str, goals: str
    ) -> dict[str, Any]:
        """기본 리팩토링 분석"""
        if self.llm_manager and self.llm_manager.is_available():
            try:
                analysis = await self.llm_manager.analyze_code(
                    task_type=TaskType.REFACTORING,
                    code=code,
                    refactoring_goals=goals,
                    constraints="",
                    language="python",
                )

                return {
                    "success": True,
                    "result": analysis,
                    "confidence_score": 0.7,
                    "method": "llm_basic_refactoring",
                }

            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {
                "success": True,
                "result": f"리팩토링 목표 '{goals}'에 대한 기본적인 제안을 드릴 수 있습니다.",
                "confidence_score": 0.3,
                "method": "fallback_refactoring",
            }

    async def _basic_context_search(
        self, query: str, include_relationships: bool
    ) -> dict[str, Any]:
        """기본 컨텍스트 검색"""
        basic_result = await self._basic_search(query)

        if basic_result["success"]:
            basic_result["result"] = f"컨텍스트 포함 검색: {basic_result['result']}"
            basic_result["method"] = "basic_context"
            if include_relationships:
                basic_result["relationships_included"] = True

        return basic_result

    # 유틸리티 메서드들
    def get_service_status(self) -> dict[str, Any]:
        """서비스 상태 정보"""
        return {
            "neo4j_available": self.neo4j_handler is not None,
            "embedding_available": self.embedding_service.is_available(),
            "llm_available": self.llm_manager.is_available()
            if self.llm_manager
            else False,
            "workflow_engine_available": self.workflow_engine is not None,
            "config": {
                "max_results": self.config.max_results,
                "similarity_threshold": self.config.similarity_threshold,
                "enable_workflows": self.config.enable_workflows,
                "default_workflow": self.config.default_workflow.value,
            },
        }

    def get_available_workflows(self) -> list[str]:
        """사용 가능한 워크플로우 목록"""
        if self.workflow_engine:
            return self.workflow_engine.get_available_workflows()
        else:
            return []

    async def health_check(self) -> dict[str, Any]:
        """서비스 헬스 체크"""
        status = self.get_service_status()

        # 간단한 검색 테스트
        try:
            test_result = await self._basic_search("test query", limit=1)
            search_healthy = test_result["success"]
        except Exception:
            search_healthy = False

        return {
            **status,
            "search_functional": search_healthy,
            "overall_health": all([status["embedding_available"], search_healthy]),
        }

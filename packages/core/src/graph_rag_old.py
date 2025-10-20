"""
LangChain/LangGraph 기반 GraphRAG 서비스
통합된 지능형 코드 분석 및 검색 시스템
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

from .neo4j_handler import Neo4jHandler
from .embedding_service import EmbeddingService
from .llm_manager import LLMManager, LLMConfig
from .graph_rag_workflow import GraphRAGWorkflow, RAGWorkflowType, RAGConfig as WorkflowRAGConfig


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
                workflow_type=self.config.default_workflow
            )
            
            return GraphRAGWorkflow(
                neo4j_handler=self.neo4j_handler,
                embedding_service=self.embedding_service,
                llm_manager=self.llm_manager,
                config=workflow_config
            )
            
        except Exception as e:
            self.logger.error(f"워크플로우 엔진 초기화 실패: {e}")
            return None
    
    async def search_similar_code(
        self,
        query: str,
        workflow_type: RAGWorkflowType = RAGWorkflowType.SIMPLE_SEARCH,
        **kwargs
    ) -> dict[str, Any]:
        """유사한 코드 검색"""
        if self.workflow_engine:
            # 워크플로우 기반 검색 
            return await self.workflow_engine.execute_workflow(query, workflow_type)
        else:
            # 기본 검색 (fallback)
            return await self._basic_search(query, **kwargs)
    
    async def analyze_code_architecture(
        self,
        query: str,
        project_name: str = None
    ) -> dict[str, Any]:
        """코드 아키텍처 분석"""
        if self.workflow_engine:
            enhanced_query = f"Analyze architecture of project {project_name}: {query}" if project_name else query
            return await self.workflow_engine.execute_workflow(
                enhanced_query,
                RAGWorkflowType.ARCHITECTURE_REVIEW
            )
        else:
            return {"success": False, "error": "워크플로우 엔진을 사용할 수 없습니다"}
    
    async def find_code_similarities(
        self,
        code_snippet: str,
        analysis_focus: str = "patterns"
    ) -> dict[str, Any]:
        """코드 유사성 분석"""
        if self.workflow_engine:
            query = f"Find similar code patterns for: {code_snippet}"
            return await self.workflow_engine.execute_workflow(
                query,
                RAGWorkflowType.CODE_SIMILARITY
            )
        else:
            return await self._basic_similarity_search(code_snippet)
    
    async def suggest_refactoring(
        self,
        target_code: str,
        refactoring_goals: str = "improve readability and maintainability"
    ) -> dict[str, Any]:
        """리팩토링 제안"""
        if self.workflow_engine:
            query = f"Suggest refactoring for code with goals: {refactoring_goals}. Code: {target_code}"
            return await self.workflow_engine.execute_workflow(
                query,
                RAGWorkflowType.REFACTORING_SUGGESTIONS
            )
        else:
            return await self._basic_refactoring_analysis(target_code, refactoring_goals)
    
    async def get_enriched_context(
        self,
        query: str,
        include_relationships: bool = True,
        **kwargs
    ) -> dict[str, Any]:
        """풍부한 컨텍스트 정보 제공"""
        if self.workflow_engine:
            return await self.workflow_engine.execute_workflow(
                query,
                RAGWorkflowType.CONTEXTUAL_ANALYSIS
            )
        else:
            return await self._basic_context_search(query, include_relationships)

    def search_similar_code(
        self,
        query: str,
        project_name: str = None,
        limit: int = None,
        similarity_threshold: float = None,
    ) -> list[dict[str, Any]]:
        """자연어 쿼리로 유사한 코드 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_service.create_query_embedding(query)

            if not query_embedding:
                self.logger.error("쿼리 임베딩 생성 실패")
                return []

            # 벡터 검색 수행
            limit = limit or self.config.max_results
            threshold = similarity_threshold or self.config.similarity_threshold

            similar_nodes = self.neo4j_handler.vector_search(
                query_embedding=query_embedding,
                limit=limit,
                similarity_threshold=threshold,
            )

            # 프로젝트 필터링 (필요한 경우)
            if project_name:
                similar_nodes = self._filter_by_project(similar_nodes, project_name)

            self.logger.info(f"유사 코드 검색 결과: {len(similar_nodes)}개")
            return similar_nodes

        except Exception as e:
            self.logger.error(f"코드 검색 실패: {e}")
            return []

    def get_enriched_context(
        self, query: str, project_name: str = None, include_relationships: bool = True
    ) -> dict[str, Any]:
        """검색 결과에 관련 컨텍스트를 포함한 풍부한 정보 제공"""
        try:
            # 기본 유사 코드 검색
            similar_nodes = self.search_similar_code(query, project_name)

            if not similar_nodes:
                return {
                    "query": query,
                    "matches": [],
                    "context": {},
                    "summary": "검색 결과가 없습니다.",
                }

            enriched_results = []

            for node in similar_nodes:
                # 기본 노드 정보
                enriched_node = {
                    "node": node,
                    "context": None,
                    "related_functions": [],
                    "usage_examples": [],
                }

                # 관련 컨텍스트 추가
                if include_relationships:
                    context = self.neo4j_handler.get_node_context(
                        node_id=node["id"], depth=self.config.context_depth
                    )
                    enriched_node["context"] = context

                    # 관련 함수들 추출
                    enriched_node["related_functions"] = (
                        self._extract_related_functions(context)
                    )

                    # 사용 예시 추출
                    enriched_node["usage_examples"] = self._extract_usage_examples(
                        context
                    )

                enriched_results.append(enriched_node)

            # 전체 컨텍스트 요약
            context_summary = self._create_context_summary(enriched_results)

            return {
                "query": query,
                "matches": enriched_results,
                "context": context_summary,
                "summary": self._create_search_summary(query, enriched_results),
            }

        except Exception as e:
            self.logger.error(f"컨텍스트 생성 실패: {e}")
            return {
                "query": query,
                "matches": [],
                "context": {},
                "summary": f"오류 발생: {str(e)}",
            }

    def find_code_dependencies(self, node_id: str) -> dict[str, Any]:
        """특정 코드 블록의 의존성 분석"""
        try:
            with self.neo4j_handler.driver.session() as session:
                query = """
                MATCH (center:CodeNode {id: $node_id})
                
                // 직접 의존성 (호출, 상속, 임포트)
                OPTIONAL MATCH (center)-[r:CALLS|INHERITS|IMPORTS]->(direct:CodeNode)
                
                // 역방향 의존성 (누가 이 노드를 사용하는가)
                OPTIONAL MATCH (reverse:CodeNode)-[rr:CALLS|INHERITS|IMPORTS]->(center)
                
                // 같은 파일 내 관련 노드들
                OPTIONAL MATCH (center)-[:CONTAINS*0..2]-(sibling:CodeNode)
                WHERE center.file_path = sibling.file_path
                
                RETURN center,
                       collect(DISTINCT {node: direct, relation: type(r)}) AS dependencies,
                       collect(DISTINCT {node: reverse, relation: type(rr)}) AS dependents,
                       collect(DISTINCT sibling) AS siblings
                """

                result = session.run(query, node_id=node_id)
                record = result.single()

                if record:
                    return {
                        "center_node": dict(record["center"]),
                        "dependencies": [
                            {
                                "node": dict(dep["node"]) if dep["node"] else None,
                                "relation_type": dep["relation"],
                            }
                            for dep in record["dependencies"]
                            if dep["node"]
                        ],
                        "dependents": [
                            {
                                "node": dict(dep["node"]) if dep["node"] else None,
                                "relation_type": dep["relation"],
                            }
                            for dep in record["dependents"]
                            if dep["node"]
                        ],
                        "siblings": [
                            dict(node)
                            for node in record["siblings"]
                            if node["id"] != node_id
                        ],
                    }
                else:
                    return {
                        "center_node": None,
                        "dependencies": [],
                        "dependents": [],
                        "siblings": [],
                    }

        except Exception as e:
            self.logger.error(f"의존성 분석 실패: {e}")
            return {
                "center_node": None,
                "dependencies": [],
                "dependents": [],
                "siblings": [],
            }

    def recommend_related_code(
        self, node_id: str, recommendation_type: str = "similar"
    ) -> list[dict[str, Any]]:
        """코드 블록과 관련된 추천 코드 제공"""
        try:
            # 현재 노드 정보 조회
            with self.neo4j_handler.driver.session() as session:
                node_query = """
                MATCH (n:CodeNode {id: $node_id})
                RETURN n
                """

                result = session.run(node_query, node_id=node_id)
                current_node = result.single()

                if not current_node:
                    return []

                current_node = dict(current_node["n"])

            recommendations = []

            if recommendation_type == "similar":
                # 유사한 코드 추천 (임베딩 기반)
                if current_node.get("source_code"):
                    similar_codes = self.search_similar_code(
                        query=current_node["source_code"], limit=5
                    )

                    # 자기 자신 제외
                    similar_codes = [
                        code for code in similar_codes if code.get("id") != node_id
                    ]

                    recommendations.extend(
                        [
                            {
                                "type": "similar_code",
                                "reason": f"유사도: {code.get('score', 0):.2f}",
                                "node": code,
                            }
                            for code in similar_codes
                        ]
                    )

            elif recommendation_type == "related":
                # 관련 코드 추천 (그래프 구조 기반)
                dependencies = self.find_code_dependencies(node_id)

                # 의존하는 코드들
                for dep in dependencies.get("dependencies", []):
                    if dep["node"]:
                        recommendations.append(
                            {
                                "type": "dependency",
                                "reason": f"의존성: {dep['relation_type']}",
                                "node": dep["node"],
                            }
                        )

                # 이 코드에 의존하는 코드들
                for dep in dependencies.get("dependents", []):
                    if dep["node"]:
                        recommendations.append(
                            {
                                "type": "dependent",
                                "reason": f"사용자: {dep['relation_type']}",
                                "node": dep["node"],
                            }
                        )

            elif recommendation_type == "contextual":
                # 컨텍스트 기반 추천 (같은 파일, 같은 클래스 등)
                dependencies = self.find_code_dependencies(node_id)

                for sibling in dependencies.get("siblings", []):
                    recommendations.append(
                        {"type": "contextual", "reason": "같은 파일", "node": sibling}
                    )

            # 중복 제거 및 점수순 정렬
            seen_ids = set()
            unique_recommendations = []

            for rec in recommendations:
                node_rec_id = rec["node"].get("id")
                if node_rec_id and node_rec_id not in seen_ids:
                    seen_ids.add(node_rec_id)
                    unique_recommendations.append(rec)

            self.logger.info(f"코드 추천 생성: {len(unique_recommendations)}개")
            return unique_recommendations[: self.config.max_results]

        except Exception as e:
            self.logger.error(f"코드 추천 실패: {e}")
            return []

    def _filter_by_project(self, nodes: list[dict], project_name: str) -> list[dict]:
        """프로젝트별 노드 필터링"""
        try:
            with self.neo4j_handler.driver.session() as session:
                # 프로젝트에 속한 노드 ID들 조회
                query = """
                MATCH (p:Project {name: $project_name})-[:CONTAINS]->(n:CodeNode)
                RETURN collect(n.id) AS node_ids
                """

                result = session.run(query, project_name=project_name)
                record = result.single()

                if record:
                    project_node_ids = set(record["node_ids"])
                    return [
                        node for node in nodes if node.get("id") in project_node_ids
                    ]
                else:
                    return []

        except Exception as e:
            self.logger.error(f"프로젝트 필터링 실패: {e}")
            return nodes

    def _extract_related_functions(self, context: dict) -> list[dict]:
        """컨텍스트에서 관련 함수들 추출"""
        related_functions = []

        for node in context.get("related_nodes", []):
            if node.get("type") in ["Function", "Method"]:
                related_functions.append(
                    {
                        "name": node.get("name"),
                        "type": node.get("type"),
                        "file_path": node.get("file_path"),
                        "docstring": node.get("docstring"),
                    }
                )

        return related_functions

    def _extract_usage_examples(self, context: dict) -> list[str]:
        """컨텍스트에서 사용 예시 추출"""
        examples = []

        for relationship in context.get("relationships", []):
            if relationship.get("context"):
                examples.append(relationship["context"])

        return examples[:3]  # 최대 3개까지만

    def _create_context_summary(self, enriched_results: list) -> dict[str, Any]:
        """전체 컨텍스트 요약 생성"""
        total_matches = len(enriched_results)
        file_paths = set()
        node_types = {}

        for result in enriched_results:
            node = result["node"]
            file_paths.add(node.get("file_path", ""))
            node_type = node.get("type", "Unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1

        return {
            "total_matches": total_matches,
            "files_involved": len(file_paths),
            "file_paths": list(file_paths),
            "node_type_distribution": node_types,
        }

    def _create_search_summary(self, query: str, results: list) -> str:
        """검색 결과 요약 생성"""
        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."

        total_matches = len(results)
        file_count = len(set(r["node"].get("file_path", "") for r in results))

        # 주요 노드 타입들
        type_counts = {}
        for result in results:
            node_type = result["node"].get("type", "Unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1

        # 가장 많은 타입
        most_common_type = (
            max(type_counts.items(), key=lambda x: x[1])[0]
            if type_counts
            else "Unknown"
        )

        summary = f"'{query}'에 대해 {total_matches}개의 매칭 코드를 {file_count}개 파일에서 발견했습니다. "
        summary += f"주요 타입: {most_common_type}"

        return summary

"""코드 검색 및 리트리버 모듈

Neo4j 그래프에서 임베딩 기반 유사도 검색과 그래프 탐색을 수행합니다.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.graph import Neo4jPersistence

logger = logging.getLogger(__name__)


class CodeSearchResult(BaseModel):
    """코드 검색 결과"""

    node_id: str = Field(..., description="노드 ID")
    name: str = Field(..., description="코드 요소 이름")
    node_type: str = Field(..., description="노드 타입 (Function, Class, etc.)")
    file_path: str = Field(..., description="파일 경로")
    source_code: str = Field(..., description="소스 코드")
    docstring: str | None = Field(None, description="문서 문자열")
    similarity_score: float = Field(..., description="유사도 점수 (0-1)")

    # 컨텍스트 정보
    related_nodes: list[dict[str, Any]] = Field(
        default_factory=list, description="연관된 노드들"
    )
    relationships: list[dict[str, Any]] = Field(
        default_factory=list, description="관계 정보"
    )

    def to_context_string(self) -> str:
        """LLM 컨텍스트용 문자열 생성

        Returns:
            포맷된 컨텍스트 문자열
        """
        context = f"### {self.node_type}: {self.name}\n"
        context += f"File: {self.file_path}\n"
        context += f"Similarity: {self.similarity_score:.2f}\n\n"

        if self.docstring:
            context += f"**Documentation:**\n{self.docstring}\n\n"

        context += f"**Code:**\n```python\n{self.source_code}\n```\n\n"

        if self.related_nodes:
            context += "**Related Components:**\n"
            for node in self.related_nodes[:5]:  # 최대 5개만
                context += (
                    f"- {node.get('type', 'Unknown')}: {node.get('name', 'Unknown')}\n"
                )

        return context


class CodeRetriever(BaseModel):
    """Neo4j에서 코드를 검색하는 리트리버

    벡터 유사도 검색과 그래프 탐색을 결합하여
    관련 코드를 효과적으로 찾아냅니다.
    """

    persistence: Neo4jPersistence = Field(..., description="Neo4j 지속성 객체")

    similarity_threshold: float = Field(default=0.5, description="유사도 임계값 (0-1)")

    max_results: int = Field(default=5, description="최대 검색 결과 수")

    context_depth: int = Field(default=2, description="그래프 탐색 깊이")

    class Config:
        arbitrary_types_allowed = True

    def search_similar_code(
        self,
        query_embedding: list[float],
        limit: int | None = None,
        threshold: float | None = None,
        expand_results: bool = True,
    ) -> list[CodeSearchResult]:
        """임베딩 유사도 기반 코드 검색 (GraphRAG 강화)

        Args:
            query_embedding: 쿼리 임베딩 벡터
            limit: 최대 결과 수 (None이면 기본값 사용)
            threshold: 유사도 임계값 (None이면 기본값 사용)
            expand_results: 그래프 탐색으로 결과 확장 여부

        Returns:
            검색 결과 리스트
        """
        limit = limit or self.max_results
        threshold = threshold or self.similarity_threshold

        try:
            # Neo4j 벡터 검색
            results = self.persistence.vector_search(
                query_embedding=query_embedding,
                limit=limit,
                similarity_threshold=threshold,
            )

            logger.info(f"✅ {len(results)}개의 유사 코드 발견")

            # 각 결과에 대해 컨텍스트 정보 가져오기
            search_results = []
            expanded_node_ids = set()  # 중복 방지

            for result in results:
                # 기본 결과 추가
                search_result = self._create_search_result(result)
                search_results.append(search_result)
                expanded_node_ids.add(result["id"])

                # 그래프 확장 검색 (MVP 최적화)
                if expand_results:
                    related_results = self._expand_search_with_graph(
                        result["id"],
                        expanded_node_ids,
                        max_expansion=2,  # MVP에서는 제한적 확장
                    )
                    search_results.extend(related_results)

            # 유사도 점수로 정렬 (확장된 결과는 낮은 점수를 가짐)
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)

            # 최종 limit 적용
            return search_results[:limit]

        except Exception as e:
            logger.error(f"❌ 코드 검색 실패: {e}")
            return []

    def get_node_context(
        self,
        node_id: str,
        depth: int | None = None,
    ) -> dict[str, Any]:
        """특정 노드의 컨텍스트 조회

        Args:
            node_id: 노드 ID
            depth: 탐색 깊이 (None이면 기본값 사용)

        Returns:
            노드 컨텍스트 정보
        """
        depth = depth or self.context_depth

        try:
            context = self.persistence.get_node_context(node_id, depth)
            logger.info(f"✅ 노드 컨텍스트 조회 완료: {node_id}")
            return context

        except Exception as e:
            logger.error(f"❌ 노드 컨텍스트 조회 실패: {e}")
            return {}

    def search_by_type(
        self,
        node_type: str,
        query_embedding: list[float] | None = None,
        limit: int | None = None,
    ) -> list[CodeSearchResult]:
        """노드 타입별 검색

        Args:
            node_type: 노드 타입 (Function, Class, etc.)
            query_embedding: 쿼리 임베딩 (None이면 타입만으로 검색)
            limit: 최대 결과 수

        Returns:
            검색 결과 리스트
        """
        # 일단 모든 유사 코드 검색
        if query_embedding:
            results = self.search_similar_code(
                query_embedding=query_embedding,
                limit=limit or self.max_results * 2,  # 필터링을 위해 더 많이 가져옴
            )

            # 타입으로 필터링
            filtered_results = [
                result
                for result in results
                if result.node_type.lower() == node_type.lower()
            ]

            # limit 적용
            return filtered_results[: limit or self.max_results]

        else:
            # 타입만으로 검색 (임베딩 없음)
            logger.warning(
                "임베딩 없이 타입만으로 검색하는 기능은 아직 구현되지 않았습니다"
            )
            return []

    def get_related_code(
        self,
        node_id: str,
        relation_types: list[str] | None = None,
        depth: int | None = None,
    ) -> list[CodeSearchResult]:
        """특정 노드와 연관된 코드 조회

        Args:
            node_id: 기준 노드 ID
            relation_types: 필터링할 관계 타입들 (None이면 모든 관계)
            depth: 탐색 깊이

        Returns:
            연관 코드 리스트
        """
        try:
            context = self.get_node_context(node_id, depth)

            related_nodes = context.get("related_nodes", [])

            # relation_types 필터링
            if relation_types and context.get("relationships"):
                # 지정된 관계 타입에 해당하는 노드만 추출
                filtered_node_ids = set()
                for rel in context["relationships"]:
                    if rel.get("type") in relation_types:
                        filtered_node_ids.add(rel.get("start_node"))
                        filtered_node_ids.add(rel.get("end_node"))

                related_nodes = [
                    node
                    for node in related_nodes
                    if node.get("id") in filtered_node_ids
                ]

            # CodeSearchResult로 변환
            results = []
            for node in related_nodes:
                result = CodeSearchResult(
                    node_id=node.get("id", ""),
                    name=node.get("name", ""),
                    node_type=node.get("type", ""),
                    file_path=node.get("file_path", ""),
                    source_code=node.get("source_code", ""),
                    docstring=node.get("docstring"),
                    similarity_score=1.0,  # 그래프 관계 기반이므로 1.0
                    related_nodes=[],
                    relationships=[],
                )
                results.append(result)

            logger.info(f"✅ {len(results)}개의 연관 코드 발견")
            return results

        except Exception as e:
            logger.error(f"❌ 연관 코드 조회 실패: {e}")
            return []

    def _create_search_result(self, result: dict[str, Any]) -> CodeSearchResult:
        """검색 결과 딕셔너리를 CodeSearchResult로 변환

        Args:
            result: Neo4j 검색 결과 딕셔너리

        Returns:
            CodeSearchResult 객체
        """
        # 컨텍스트 조회
        context = self.persistence.get_node_context(
            result["id"], depth=self.context_depth
        )

        return CodeSearchResult(
            node_id=result["id"],
            name=result["name"],
            node_type=result["type"],
            file_path=result["file_path"],
            source_code=result["source_code"],
            docstring=result.get("docstring"),
            similarity_score=result["score"],
            related_nodes=context.get("related_nodes", []),
            relationships=context.get("relationships", []),
        )

    def _expand_search_with_graph(
        self, node_id: str, expanded_node_ids: set[str], max_expansion: int = 2
    ) -> list[CodeSearchResult]:
        """그래프 탐색으로 검색 결과 확장

        Args:
            node_id: 확장할 기준 노드 ID
            expanded_node_ids: 이미 포함된 노드 ID들
            max_expansion: 최대 확장 개수

        Returns:
            확장된 검색 결과 리스트
        """
        try:
            expanded_results = []

            # 직접 연결된 노드들 가져오기
            context = self.persistence.get_node_context(node_id, depth=1)
            related_nodes = context.get("related_nodes", [])

            # 관련성이 높은 노드들 선별 (함수 호출, 클래스 상속 등)
            high_priority_relations = ["CALLS", "INHERITS", "CONTAINS", "IMPORTS"]
            relationships = context.get("relationships", [])

            priority_nodes = []
            for rel in relationships:
                if rel.get("type") in high_priority_relations:
                    target_node_id = (
                        rel.get("end_node")
                        if rel.get("start_node") == node_id
                        else rel.get("start_node")
                    )
                    if target_node_id and target_node_id not in expanded_node_ids:
                        # 해당 노드 정보 찾기
                        target_node = next(
                            (n for n in related_nodes if n.get("id") == target_node_id),
                            None,
                        )
                        if target_node:
                            priority_nodes.append(target_node)

            # 최대 확장 개수까지만 추가
            for node in priority_nodes[:max_expansion]:
                expanded_result = CodeSearchResult(
                    node_id=node.get("id", ""),
                    name=node.get("name", ""),
                    node_type=node.get("type", ""),
                    file_path=node.get("file_path", ""),
                    source_code=node.get("source_code", ""),
                    docstring=node.get("docstring"),
                    similarity_score=0.5,  # 확장된 결과는 낮은 점수
                    related_nodes=[],
                    relationships=[],
                )
                expanded_results.append(expanded_result)
                expanded_node_ids.add(node.get("id", ""))

            return expanded_results

        except Exception as e:
            logger.error(f"❌ 그래프 확장 검색 실패: {e}")
            return []

    def search_with_hybrid_approach(
        self,
        query_embedding: list[float],
        query_text: str,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
    ) -> list[CodeSearchResult]:
        """하이브리드 검색: 벡터 유사도 + 그래프 구조

        Args:
            query_embedding: 쿼리 임베딩 벡터
            query_text: 쿼리 텍스트 (키워드 매칭용)
            vector_weight: 벡터 유사도 가중치
            graph_weight: 그래프 구조 가중치

        Returns:
            하이브리드 점수로 정렬된 검색 결과
        """
        try:
            # 1. 벡터 유사도 검색
            vector_results = self.search_similar_code(
                query_embedding,
                limit=self.max_results * 2,  # 더 많이 가져와서 재순위
                expand_results=False,
            )

            # 2. 키워드 기반 텍스트 검색 결과도 포함
            text_matched_results = self._search_by_text_similarity(query_text)

            # 3. 하이브리드 점수 계산
            all_results = {}

            # 벡터 검색 결과 추가
            for result in vector_results:
                all_results[result.node_id] = result
                result.similarity_score = result.similarity_score * vector_weight

            # 텍스트 검색 결과 추가/업데이트
            for result in text_matched_results:
                if result.node_id in all_results:
                    # 기존 점수에 텍스트 점수 추가
                    all_results[result.node_id].similarity_score += (
                        result.similarity_score * graph_weight
                    )
                else:
                    # 새로운 결과 추가
                    result.similarity_score *= graph_weight
                    all_results[result.node_id] = result

            # 4. 최종 점수로 정렬
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: x.similarity_score, reverse=True)

            return final_results[: self.max_results]

        except Exception as e:
            logger.error(f"❌ 하이브리드 검색 실패: {e}")
            return []

    def _search_by_text_similarity(self, query_text: str) -> list[CodeSearchResult]:
        """텍스트 유사도 기반 검색 (키워드 매칭)

        Args:
            query_text: 검색할 텍스트

        Returns:
            텍스트 매칭 결과
        """
        # MVP에서는 간단한 키워드 매칭만 구현
        # 추후 Neo4j의 full-text search 기능 활용 가능
        try:
            keywords = query_text.lower().split()
            if not keywords:
                return []

            # 각 키워드로 노드 이름이나 소스 코드에서 매칭되는 노드 찾기
            # 실제로는 Neo4j의 텍스트 검색 기능을 사용하는 것이 좋지만
            # MVP에서는 간단히 구현
            return []  # 현재 버전에서는 벡터 검색에만 집중

        except Exception as e:
            logger.error(f"❌ 텍스트 검색 실패: {e}")
            return []

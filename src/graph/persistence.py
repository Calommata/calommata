"""
Neo4j 데이터베이스 지속성 계층

코드 그래프의 저장, 조회, 벡터 인덱스 관리를 담당합니다.
Graph 패키지의 모델 데이터를 Neo4j에 저장하고 검색하는 기능을 제공합니다.

리팩토링된 버전:
- 쿼리 분리 (queries.py)
- 예외 처리 개선 (exceptions.py)
- 배치 처리 최적화
- 타입 힌팅 강화
- 로깅 개선
"""

import logging
import os
from typing import Any

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from .exceptions import (
    ConnectionError as PersistenceConnectionError,
    IndexCreationError,
    InvalidDataError,
    NodeNotFoundError,
)
from .models import CodeGraph, CodeNode, CodeRelation
from .queries import Neo4jQueries

logger = logging.getLogger(__name__)


class Neo4jPersistence:
    """Neo4j 데이터베이스 연결 및 그래프 지속성 관리

    리팩토링된 버전:
    - 배치 크기 설정 가능
    - 타입 안전성 강화
    - 에러 처리 개선
    """

    # 기본 배치 크기
    DEFAULT_BATCH_SIZE = 500

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Neo4j 지속성 계층 초기화

        Args:
            uri: Neo4j 데이터베이스 URI (환경변수 NEO4J_URI 사용 가능)
            user: 사용자명 (환경변수 NEO4J_USER 사용 가능)
            password: 패스워드 (환경변수 NEO4J_PASSWORD 사용 가능)
            batch_size: 배치 처리 크기 (기본값: 500)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.batch_size = batch_size

        self._driver: Driver | None = None
        self.logger = logger
        self._queries = Neo4jQueries()

    @property
    def driver(self) -> Driver:
        """드라이버 속성 - 연결되지 않았으면 예외 발생"""
        if self._driver is None:
            raise PersistenceConnectionError("데이터베이스에 연결되지 않음")
        return self._driver

    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._driver is not None

    def connect(self) -> bool:
        """Neo4j 데이터베이스에 연결

        Returns:
            bool: 연결 성공 여부

        Raises:
            PersistenceConnectionError: 연결 실패 시
        """
        try:
            # SSL 인증서 검증을 우회하기 위한 설정
            import ssl

            # SSL 컨텍스트 생성 (인증서 검증 비활성화)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # URI 스킴에 따른 연결 설정
            if self.uri.startswith("neo4j+s://") or self.uri.startswith("bolt+s://"):
                # SSL 사용하는 경우 SSL 컨텍스트 적용
                uri_without_ssl = self.uri.replace("neo4j+s://", "neo4j://").replace(
                    "bolt+s://", "bolt://"
                )
                self._driver = GraphDatabase.driver(
                    uri_without_ssl,
                    auth=(self.user, self.password),
                    encrypted=True,
                    ssl_context=ssl_context,
                )
            else:
                # 일반적인 연결
                self._driver = GraphDatabase.driver(
                    self.uri, auth=(self.user, self.password)
                )
            # 연결 테스트
            with self._driver.session() as session:
                session.run(self._queries.TEST_CONNECTION)

            self.logger.info(f"✅ Neo4j 연결 성공: {self.uri}")
            return True

        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"❌ Neo4j 연결 실패: {e}")
            self._driver = None
            raise PersistenceConnectionError(f"Neo4j 연결 실패: {e}") from e
        except Exception as e:
            self.logger.error(f"❌ 예상치 못한 오류: {e}")
            self._driver = None
            raise PersistenceConnectionError(f"예상치 못한 연결 오류: {e}") from e

    def close(self) -> None:
        """연결 종료"""
        if self._driver:
            try:
                self._driver.close()
                self.logger.info("✅ Neo4j 연결 종료")
            except Exception as e:
                self.logger.warning(f"⚠️ 연결 종료 중 오류: {e}")
            finally:
                self._driver = None

    def create_constraints_and_indexes(self) -> bool:
        """데이터베이스 제약 조건 및 인덱스 생성

        Returns:
            bool: 성공 여부

        Raises:
            IndexCreationError: 인덱스 생성 실패 시
        """
        try:
            with self.driver.session() as session:
                # 제약 조건 생성
                for constraint in self._queries.CONSTRAINTS:
                    self._execute_schema_query(session, constraint, "제약 조건")

                # 기본 인덱스 생성
                for index in self._queries.INDEXES:
                    self._execute_schema_query(session, index, "인덱스")

                # 벡터 인덱스 생성 (기본적으로 768차원 사용)
                self._execute_schema_query(
                    session, self._queries.VECTOR_INDEX, "벡터 인덱스"
                )

            self.logger.info("✅ 모든 제약 조건 및 인덱스 생성 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 제약 조건/인덱스 생성 실패: {e}")
            raise IndexCreationError(f"인덱스 생성 실패: {e}") from e

    def create_vector_index_for_dimension(self, dimension: int) -> bool:
        """특정 차원에 맞는 벡터 인덱스 생성

        Args:
            dimension: 임베딩 벡터 차원 수

        Returns:
            bool: 성공 여부
        """
        try:
            with self.driver.session() as session:
                # 기존 벡터 인덱스 삭제
                try:
                    session.run("DROP INDEX code_embedding_index IF EXISTS")
                    session.run("DROP INDEX code_embedding_index_hf IF EXISTS")
                except Exception:
                    pass  # 인덱스가 없으면 무시

                # 새로운 벡터 인덱스 생성
                vector_index_query = f"""
                    CREATE VECTOR INDEX code_embedding_index IF NOT EXISTS 
                    FOR (n:CodeNode) ON (n.embedding) 
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """

                self._execute_schema_query(
                    session, vector_index_query, f"{dimension}차원 벡터 인덱스"
                )

            self.logger.info(f"✅ {dimension}차원 벡터 인덱스 생성 완료")
            return True

        except Exception as e:
            self.logger.error(f"❌ 벡터 인덱스 생성 실패: {e}")
            return False

    def _execute_schema_query(self, session: Any, query: str, description: str) -> None:
        """스키마 쿼리 실행 헬퍼 메서드

        Args:
            session: Neo4j 세션
            query: 실행할 쿼리
            description: 쿼리 설명 (로깅용)
        """
        try:
            session.run(query)
            self.logger.info(f"✅ {description} 생성: {query[:50]}...")
        except Exception as e:
            self.logger.warning(f"⚠️ {description} 생성 실패 (이미 존재할 수 있음): {e}")

    def save_code_graph(
        self, graph: CodeGraph, project_name: str | None = None
    ) -> bool:
        """전체 코드 그래프를 Neo4j에 저장

        Args:
            graph: 저장할 CodeGraph 객체
            project_name: 프로젝트 이름 (None이면 graph.project_name 사용)

        Returns:
            bool: 저장 성공 여부

        Raises:
            InvalidDataError: 유효하지 않은 데이터
        """
        project_name = project_name or graph.project_name

        if not project_name:
            raise InvalidDataError("프로젝트 이름이 필요합니다")

        try:
            # 1단계: 프로젝트 정보 저장
            self._save_project_info(graph, project_name)

            # 2단계: 노드 저장 (배치 처리)
            self._save_code_nodes_batch(list(graph.nodes.values()), project_name)

            # 3단계: 관계 저장 (배치 처리)
            self._save_code_relations_batch(graph.relations)

            self.logger.info(
                f"✅ 그래프 저장 완료: {len(graph.nodes)}개 노드, "
                f"{len(graph.relations)}개 관계"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ 그래프 저장 실패: {e}")
            raise

    def _save_project_info(self, graph: CodeGraph, project_name: str) -> None:
        """프로젝트 정보 저장 (내부 메서드)

        Raises:
            QueryExecutionError: 쿼리 실행 실패 시
        """
        try:
            with self.driver.session() as session:
                stats = graph.get_statistics()
                session.run(
                    self._queries.MERGE_PROJECT,
                    name=project_name,
                    path=graph.project_path,
                    total_files=graph.total_files,
                    total_lines=graph.total_lines,
                    total_nodes=stats["total_nodes"],
                    total_relations=stats["total_relations"],
                    analysis_version=graph.analysis_version,
                    created_at=graph.created_at.isoformat(),
                )
                self.logger.info(f"✅ 프로젝트 정보 저장: {project_name}")

        except Exception as e:
            self.logger.error(f"❌ 프로젝트 정보 저장 실패: {e}")
            raise

    def _save_code_nodes_batch(self, nodes: list[CodeNode], project_name: str) -> None:
        """코드 노드들을 배치로 저장 (내부 메서드)

        Args:
            nodes: 저장할 노드 리스트
            project_name: 프로젝트 이름

        Raises:
            QueryExecutionError: 쿼리 실행 실패 시
        """
        if not nodes:
            self.logger.warning("⚠️ 저장할 노드가 없습니다")
            return

        try:
            # 배치 처리로 성능 최적화
            total_batches = (len(nodes) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(nodes))
                batch = nodes[start_idx:end_idx]

                # Neo4j 형식으로 변환
                neo4j_nodes = [node.to_neo4j_node() for node in batch]

                with self.driver.session() as session:
                    session.run(
                        self._queries.MERGE_CODE_NODES_BATCH,
                        nodes=neo4j_nodes,
                        project_name=project_name,
                    )

                self.logger.info(
                    f"✅ 노드 배치 {batch_idx + 1}/{total_batches} 저장 완료 "
                    f"({len(batch)}개)"
                )

            self.logger.info(f"✅ 총 {len(nodes)}개 코드 노드 저장 완료")

        except Exception as e:
            self.logger.error(f"❌ 코드 노드 배치 저장 실패: {e}")
            raise

    def _save_code_relations_batch(self, relations: list[CodeRelation]) -> None:
        """코드 관계들을 배치로 저장 (내부 메서드)

        Args:
            relations: 저장할 관계 리스트

        Raises:
            QueryExecutionError: 쿼리 실행 실패 시
        """
        if not relations:
            self.logger.warning("⚠️ 저장할 관계가 없습니다")
            return

        try:
            # 관계 타입별로 그룹화하여 배치 처리
            relations_by_type: dict[str, list[CodeRelation]] = {}
            for rel in relations:
                rel_type = (
                    rel.relation_type.value
                    if hasattr(rel.relation_type, "value")
                    else str(rel.relation_type)
                )
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                relations_by_type[rel_type].append(rel)

            # 각 타입별로 배치 저장
            total_saved = 0
            for rel_type, type_relations in relations_by_type.items():
                self._save_relations_of_type(rel_type, type_relations)
                total_saved += len(type_relations)

            self.logger.info(f"✅ 총 {total_saved}개 코드 관계 저장 완료")

        except Exception as e:
            self.logger.error(f"❌ 코드 관계 배치 저장 실패: {e}")
            raise

    def _save_relations_of_type(
        self, relation_type: str, relations: list[CodeRelation]
    ) -> None:
        """특정 타입의 관계들을 배치로 저장

        Args:
            relation_type: 관계 타입
            relations: 해당 타입의 관계 리스트
        """
        total_batches = (len(relations) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(relations))
            batch = relations[start_idx:end_idx]

            # 쿼리 생성 및 실행
            query = self._queries.create_relation_query(relation_type)

            with self.driver.session() as session:
                for rel in batch:
                    # 동적으로 생성된 쿼리이므로 타입 체크 무시
                    session.run(
                        query,  # type: ignore[arg-type]
                        from_id=rel.from_node_id,
                        to_id=rel.to_node_id,
                        weight=rel.weight,
                        line_number=rel.line_number,
                        context=rel.context,
                        created_at=rel.created_at.isoformat(),
                    )

            self.logger.info(
                f"✅ {relation_type} 관계 배치 {batch_idx + 1}/{total_batches} "
                f"저장 완료 ({len(batch)}개)"
            )

    def update_node_embedding(
        self, node_id: str, embedding: list[float], model: str
    ) -> bool:
        """노드의 임베딩 벡터 업데이트

        Args:
            node_id: 업데이트할 노드 ID
            embedding: 임베딩 벡터
            model: 임베딩 모델명

        Returns:
            bool: 성공 여부

        Raises:
            NodeNotFoundError: 노드를 찾을 수 없을 때
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    self._queries.UPDATE_NODE_EMBEDDING,
                    node_id=node_id,
                    embedding=embedding,
                    model=model,
                )

                if result.single():
                    self.logger.info(f"✅ 임베딩 업데이트: {node_id}")
                    return True
                else:
                    raise NodeNotFoundError(f"노드를 찾을 수 없음: {node_id}")

        except NodeNotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"❌ 임베딩 업데이트 실패: {e}")
            raise

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        project_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """벡터 유사도 기반 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            limit: 반환할 최대 결과 수
            similarity_threshold: 유사도 임계값 (0.0 ~ 1.0)

        Returns:
            list[dict]: 유사한 노드들의 정보 리스트
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    self._queries.VECTOR_SEARCH,
                    query_embedding=query_embedding,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    project_name=project_name,
                )

                results = [record.data() for record in result]
                self.logger.info(f"✅ 벡터 검색 완료: {len(results)}개 결과")
                return results

        except Exception as e:
            self.logger.error(f"❌ 벡터 검색 실패: {e}")
            return []

    def get_node_context(self, node_id: str, depth: int = 2) -> dict[str, Any]:
        """노드와 주변 컨텍스트 조회

        Args:
            node_id: 조회할 노드 ID
            depth: 주변 관계 깊이 (기본값: 2)

        Returns:
            dict: 노드 정보와 관련 노드들
        """
        try:
            with self.driver.session() as session:
                # 깊이에 따른 다른 쿼리 사용 (타입 안전성을 위해)
                if depth == 1:
                    query = """
                        MATCH (center:CodeNode {id: $node_id})
                        OPTIONAL MATCH path = (center)-[*1..1]-(related:CodeNode)
                        WITH center, collect(DISTINCT related) AS related_nodes,
                             [rel IN collect(DISTINCT relationships(path)) WHERE rel IS NOT NULL | rel] AS all_relationship_lists
                        
                        RETURN center,
                               related_nodes,
                               [rel IN reduce(flat = [], list IN all_relationship_lists | flat + list) WHERE rel IS NOT NULL | {
                                   type: type(rel),
                                   start_node: startNode(rel).id,
                                   end_node: endNode(rel).id,
                                   properties: properties(rel)
                               }] AS relationships
                    """
                elif depth == 3:
                    query = """
                        MATCH (center:CodeNode {id: $node_id})
                        OPTIONAL MATCH path = (center)-[*1..3]-(related:CodeNode)
                        WITH center, collect(DISTINCT related) AS related_nodes,
                             [rel IN collect(DISTINCT relationships(path)) WHERE rel IS NOT NULL | rel] AS all_relationship_lists
                        
                        RETURN center,
                               related_nodes,
                               [rel IN reduce(flat = [], list IN all_relationship_lists | flat + list) WHERE rel IS NOT NULL | {
                                   type: type(rel),
                                   start_node: startNode(rel).id,
                                   end_node: endNode(rel).id,
                                   properties: properties(rel)
                               }] AS relationships
                    """
                else:  # depth == 2 (기본값)
                    query = """
                        MATCH (center:CodeNode {id: $node_id})
                        OPTIONAL MATCH path = (center)-[*1..2]-(related:CodeNode)
                        WITH center, collect(DISTINCT related) AS related_nodes,
                             [rel IN collect(DISTINCT relationships(path)) WHERE rel IS NOT NULL | rel] AS all_relationship_lists
                        
                        RETURN center,
                               related_nodes,
                               [rel IN reduce(flat = [], list IN all_relationship_lists | flat + list) WHERE rel IS NOT NULL | {
                                   type: type(rel),
                                   start_node: startNode(rel).id,
                                   end_node: endNode(rel).id,
                                   properties: properties(rel)    
                               }] AS relationships
                    """

                result = session.run(query, node_id=node_id)
                record = result.single()

                if record:
                    # 관계를 필터링하여 None 제거
                    relationships = [
                        rel for rel in record["relationships"] if rel is not None
                    ]

                    return {
                        "center_node": dict(record["center"]),
                        "related_nodes": [
                            dict(node) for node in record["related_nodes"]
                        ],
                        "relationships": relationships,
                    }
                else:
                    self.logger.warning(f"⚠️ 노드를 찾을 수 없음: {node_id}")
                    return {
                        "center_node": None,
                        "related_nodes": [],
                        "relationships": [],
                    }

        except Exception as e:
            self.logger.error(f"❌ 노드 컨텍스트 조회 실패: {e}")
            return {"center_node": None, "related_nodes": [], "relationships": []}

    def get_database_statistics(self) -> dict[str, Any]:
        """데이터베이스 통계 정보 조회

        Returns:
            dict: 노드, 관계, 타입별 통계
        """
        try:
            with self.driver.session() as session:
                # 노드 수 조회
                node_stats = session.run(self._queries.GET_NODE_STATS).data()

                # 관계 수 조회
                rel_stats = session.run(self._queries.GET_RELATION_STATS).data()

                # 전체 통계
                total_stats_record = session.run(self._queries.GET_TOTAL_STATS).single()

                if not total_stats_record:
                    self.logger.warning("⚠️ 통계 데이터를 찾을 수 없음")
                    return {}

                return {
                    "total_nodes": total_stats_record["total_nodes"] or 0,
                    "total_relationships": total_stats_record["total_relationships"]
                    or 0,
                    "node_types": {
                        stat["type"]: stat["count"]
                        for stat in node_stats
                        if stat["type"]
                    },
                    "relation_types": {
                        stat["type"]: stat["count"]
                        for stat in rel_stats
                        if stat["type"]
                    },
                }

        except Exception as e:
            self.logger.error(f"❌ 통계 조회 실패: {e}")
            return {}

    def clear_project_data(self, project_name: str) -> bool:
        """프로젝트 데이터 삭제

        Args:
            project_name: 삭제할 프로젝트명

        Returns:
            bool: 성공 여부
        """
        try:
            with self.driver.session() as session:
                session.run(self._queries.DELETE_PROJECT, project_name=project_name)
                self.logger.info(f"✅ 프로젝트 데이터 삭제: {project_name}")
                return True

        except Exception as e:
            self.logger.error(f"❌ 프로젝트 데이터 삭제 실패: {e}")
            return False

    def __enter__(self) -> "Neo4jPersistence":
        """컨텍스트 매니저 진입

        Returns:
            Neo4jPersistence: 자기 자신
        """
        try:
            self.connect()
        except Exception:
            # connect 메서드에서 이미 예외를 발생시키므로 그대로 전파
            raise
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """컨텍스트 매니저 종료

        Args:
            exc_type: 예외 타입
            exc_val: 예외 값
            exc_tb: 예외 트레이스백
        """
        self.close()

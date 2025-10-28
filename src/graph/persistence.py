"""Neo4j 데이터베이스 지속성 계층 (오케스트레이터)

그래프 저장, 검색, 통계 조회를 조합합니다.
"""

import logging

from neo4j import Driver

from .models import CodeGraph
from .connection_manager import ConnectionManager
from .node_persistence import NodePersistence
from .relationship_persistence import RelationshipPersistence
from .vector_search_manager import VectorSearchManager
from .statistics_manager import StatisticsManager
from .exceptions import InvalidDataError

logger = logging.getLogger(__name__)


class Neo4jPersistence:
    """Neo4j 지속성 오케스트레이터

    리팩토링된 버전:
    - ConnectionManager: DB 연결 관리
    - NodePersistence: 노드 저장/조회
    - RelationshipPersistence: 관계 저장
    - VectorSearchManager: 벡터 검색
    - StatisticsManager: 통계/프로젝트 관리
    """

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
        self._conn_manager = ConnectionManager(uri, user, password)
        self.batch_size = batch_size

        # 컴포넌트들은 나중에 초기화됨 (driver가 필요하므로)
        self._node_persistence: NodePersistence | None = None
        self._relation_persistence: RelationshipPersistence | None = None
        self._vector_search: VectorSearchManager | None = None
        self._statistics: StatisticsManager | None = None

    @property
    def driver(self) -> Driver:
        """드라이버 속성"""
        return self._conn_manager.driver

    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._conn_manager.is_connected

    @property
    def _node_mgr(self) -> NodePersistence:
        """노드 관리자 (Lazy initialization)"""
        if self._node_persistence is None:
            self._node_persistence = NodePersistence(self.driver, self.batch_size)
        return self._node_persistence

    @property
    def _rel_mgr(self) -> RelationshipPersistence:
        """관계 관리자 (Lazy initialization)"""
        if self._relation_persistence is None:
            self._relation_persistence = RelationshipPersistence(
                self.driver, self.batch_size
            )
        return self._relation_persistence

    @property
    def _vector_mgr(self) -> VectorSearchManager:
        """벡터 검색 관리자 (Lazy initialization)"""
        if self._vector_search is None:
            self._vector_search = VectorSearchManager(self.driver)
        return self._vector_search

    @property
    def _stats_mgr(self) -> StatisticsManager:
        """통계 관리자 (Lazy initialization)"""
        if self._statistics is None:
            self._statistics = StatisticsManager(self.driver)
        return self._statistics

    def connect(self) -> bool:
        """Neo4j 데이터베이스에 연결"""
        return self._conn_manager.connect()

    def close(self) -> None:
        """연결 종료"""
        self._conn_manager.close()

    def create_constraints_and_indexes(self) -> bool:
        """데이터베이스 제약 조건 및 인덱스 생성"""
        return self._conn_manager.create_constraints_and_indexes()

    def create_vector_index_for_dimension(self, dimension: int) -> bool:
        """특정 차원에 맞는 벡터 인덱스 생성"""
        return self._conn_manager.create_vector_index_for_dimension(dimension)

    def save_code_graph(
        self, graph: CodeGraph, project_name: str | None = None
    ) -> bool:
        """전체 코드 그래프를 Neo4j에 저장

        Args:
            graph: 저장할 CodeGraph 객체
            project_name: 프로젝트 이름 (None이면 graph.project_name 사용)

        Returns:
            bool: 저장 성공 여부
        """
        project_name = project_name or graph.project_name

        if not project_name:
            raise InvalidDataError("프로젝트 이름이 필요합니다")

        try:
            # 1단계: 프로젝트 정보 저장
            self._node_mgr.save_project_info(graph, project_name)

            # 2단계: 노드 저장
            self._node_mgr.save_nodes_batch(list(graph.nodes.values()), project_name)

            # 3단계: 관계 저장
            self._rel_mgr.save_relations_batch(graph.relations)

            logger.info(
                f"✅ 그래프 저장 완료: {len(graph.nodes)}개 노드, "
                f"{len(graph.relations)}개 관계"
            )
            return True

        except Exception as e:
            logger.error(f"❌ 그래프 저장 실패: {e}")
            raise

    def update_node_embedding(
        self, node_id: str, embedding: list[float], model: str
    ) -> bool:
        """노드의 임베딩 벡터 업데이트"""
        return self._node_mgr.update_embedding(node_id, embedding, model)

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        project_name: str | None = None,
    ) -> list[dict]:
        """벡터 유사도 기반 검색"""
        return self._vector_mgr.vector_search(
            query_embedding, limit, similarity_threshold, project_name
        )

    def get_node_context(self, node_id: str, depth: int = 2) -> dict:
        """노드와 주변 컨텍스트 조회"""
        return self._vector_mgr.get_node_context(node_id, depth)

    def get_database_statistics(self) -> dict:
        """프로젝트 통계 정보 조회"""
        return self._stats_mgr.get_statistics()

    def clear_project_data(self, project_name: str) -> bool:
        """프로젝트 데이터 삭제"""
        return self._stats_mgr.clear_project_data(project_name)

    def __enter__(self) -> "Neo4jPersistence":
        """컨텍스트 매니저 진입"""
        try:
            self.connect()
        except Exception:
            raise
        return self

    def __exit__(self, *args) -> None:
        """컨텍스트 매니저 종료"""
        self.close()

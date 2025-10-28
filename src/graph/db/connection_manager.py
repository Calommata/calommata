"""Neo4j 연결 관리 모듈

데이터베이스 연결, 세션 관리, 스키마 생성을 담당합니다.
"""

import logging
import os
import ssl

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.graph.db.error import (
    ConnectionError as PersistenceConnectionError,
    IndexCreationError,
)
from src.graph.db.queries import Neo4jQueries

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Neo4j 연결 및 스키마 관리"""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """초기화

        Args:
            uri: Neo4j URI (환경변수 NEO4J_URI 사용 가능)
            user: 사용자명 (환경변수 NEO4J_USER 사용 가능)
            password: 패스워드 (환경변수 NEO4J_PASSWORD 사용 가능)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

        self._driver: Driver | None = None
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
            # SSL 컨텍스트 생성 (인증서 검증 비활성화)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # URI 스킴에 따른 연결 설정
            if self.uri.startswith("neo4j+s://") or self.uri.startswith("bolt+s://"):
                # SSL 사용하는 경우
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

            logger.info(f"✅ Neo4j 연결 성공: {self.uri}")
            return True

        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"❌ Neo4j 연결 실패: {e}")
            self._driver = None
            raise PersistenceConnectionError(f"Neo4j 연결 실패: {e}") from e
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류: {e}")
            self._driver = None
            raise PersistenceConnectionError(f"예상치 못한 연결 오류: {e}") from e

    def close(self) -> None:
        """연결 종료"""
        if self._driver:
            try:
                self._driver.close()
                logger.info("✅ Neo4j 연결 종료")
            except Exception as e:
                logger.warning(f"⚠️ 연결 종료 중 오류: {e}")
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

                # 벡터 인덱스 생성
                self._execute_schema_query(
                    session, self._queries.VECTOR_INDEX, "벡터 인덱스"
                )

            logger.info("✅ 모든 제약 조건 및 인덱스 생성 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 제약 조건/인덱스 생성 실패: {e}")
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
                except Exception:
                    pass

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

            logger.info(f"✅ {dimension}차원 벡터 인덱스 생성 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 벡터 인덱스 생성 실패: {e}")
            return False

    @staticmethod
    def _execute_schema_query(session: Session, query: str, description: str) -> None:
        """스키마 쿼리 실행 헬퍼 메서드

        Args:
            session: Neo4j 세션
            query: 실행할 쿼리
            description: 쿼리 설명 (로깅용)
        """
        try:
            session.run(query)  # type: ignore[arg-type]
            logger.info(f"✅ {description} 생성: {query[:50]}...")
        except Exception as e:
            logger.warning(f"⚠️ {description} 생성 실패 (이미 존재할 수 있음): {e}")

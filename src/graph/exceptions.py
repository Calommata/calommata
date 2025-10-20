"""Neo4j Persistence 계층 예외 정의

데이터베이스 작업 중 발생할 수 있는 예외들을 정의합니다.
"""


class PersistenceError(Exception):
    """지속성 계층 기본 예외"""

    pass


class ConnectionError(PersistenceError):
    """데이터베이스 연결 오류"""

    pass


class QueryExecutionError(PersistenceError):
    """쿼리 실행 오류"""

    pass


class NodeNotFoundError(PersistenceError):
    """노드를 찾을 수 없음"""

    pass


class InvalidDataError(PersistenceError):
    """유효하지 않은 데이터"""

    pass


class IndexCreationError(PersistenceError):
    """인덱스 생성 오류"""

    pass

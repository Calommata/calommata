from .persistence_error import PersistenceError


class ConnectionError(PersistenceError):
    """데이터베이스 연결 오류"""

    pass

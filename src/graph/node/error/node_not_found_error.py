from src.graph.db.error.persistence_error import PersistenceError


class NodeNotFoundError(PersistenceError):
    """노드를 찾을 수 없음"""

    pass

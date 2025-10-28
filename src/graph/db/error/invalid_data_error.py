from .persistence_error import PersistenceError


class InvalidDataError(PersistenceError):
    """유효하지 않은 데이터"""

    pass

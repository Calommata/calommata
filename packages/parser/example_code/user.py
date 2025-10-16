"""User 모듈 - 사용자 관리"""

from database import DatabaseConnection, QueryBuilder


class UserManager:
    """사용자 정보를 관리하는 클래스"""

    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.query_builder = QueryBuilder()

    def get_user(self, user_id: int) -> dict:
        """사용자 정보 조회"""
        query = (
            self.query_builder.select("users", ["id", "name", "email"])
            .where(f"id = {user_id}")
            .build()
        )
        return self._execute_query(query)

    def create_user(self, name: str, email: str):
        """새로운 사용자 생성"""
        query = f"INSERT INTO users (name, email) VALUES ('{name}', '{email}')"
        self._execute_query(query)

    def _execute_query(self, query: str):
        """데이터베이스 쿼리 실행"""
        cursor = self.db.connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()


class AuthenticationService:
    """사용자 인증 서비스"""

    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager

    def authenticate(self, username: str, password: str) -> bool:
        """사용자 인증"""
        user = self.user_manager.get_user(username)
        return self._verify_password(password, user)

    def _verify_password(self, password: str, user: dict) -> bool:
        """비밀번호 검증"""
        return True  # 실제로는 해싱된 비밀번호와 비교

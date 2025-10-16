"""Database 모듈 - 데이터베이스 연결 및 쿼리 관리"""

import sqlite3
from typing import List


class DatabaseConnection:
    """데이터베이스 연결을 관리하는 클래스"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """데이터베이스에 연결"""
        self.connection = sqlite3.connect(self.db_path)

    def close(self):
        """데이터베이스 연결 종료"""
        if self.connection:
            self.connection.close()


class QueryBuilder:
    """SQL 쿼리를 생성하는 클래스"""

    def __init__(self):
        self.query = ""

    def select(self, table: str, columns: List[str] = None) -> "QueryBuilder":
        """SELECT 쿼리 시작"""
        cols = ", ".join(columns) if columns else "*"
        self.query = f"SELECT {cols} FROM {table}"
        return self

    def where(self, condition: str) -> "QueryBuilder":
        """WHERE 조건 추가"""
        self.query += f" WHERE {condition}"
        return self

    def build(self) -> str:
        """쿼리 완성"""
        return self.query

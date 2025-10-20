"""테스트 설정"""

import pytest
import os


@pytest.fixture
def mock_env_vars(monkeypatch):
    """테스트용 환경 변수 설정"""
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")


@pytest.fixture
def sample_code():
    """테스트용 샘플 코드"""
    return """
def calculate_sum(a: int, b: int) -> int:
    '''두 숫자의 합을 계산합니다.
    
    Args:
        a: 첫 번째 숫자
        b: 두 번째 숫자
        
    Returns:
        두 숫자의 합
    '''
    return a + b


class Calculator:
    '''계산기 클래스'''
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        '''덧셈 수행'''
        result = calculate_sum(a, b)
        self.history.append(('add', a, b, result))
        return result
"""

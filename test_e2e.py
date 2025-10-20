"""E2E 테스트 케이스

간단한 파이썬 코드를 분석하고 GraphRAG 검색이 잘 동작하는지 테스트합니다.
Gemini 무료 플랜 제한을 고려하여 최소한의 테스트만 수행합니다.
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.core import CoreConfig, create_from_config

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_config() -> CoreConfig:
    """테스트용 설정"""
    config = CoreConfig()

    # HuggingFace 임베딩 사용
    config.embedding.provider = "huggingface"
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding.device = "cpu"

    # Gemini API 키 확인 (환경변수에서)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다")

    config.llm.api_key = api_key
    config.llm.model_name = "gemini-2.0-flash-lite"
    config.llm.temperature = 0.1
    config.llm.max_tokens = 1024  # 작게 설정하여 무료 플랜 보호

    # Neo4j 로컬 설정
    config.neo4j.uri = "bolt://localhost:7687"
    config.neo4j.user = "neo4j"
    config.neo4j.password = "password"

    # 테스트용 프로젝트명
    config.project_name = "test-project"

    return config


@pytest.fixture
def sample_code_files(tmp_path: Path) -> Path:
    """테스트용 샘플 코드 파일들 생성"""

    # 간단한 파이썬 모듈 생성
    main_py = tmp_path / "main.py"
    main_py.write_text(
        '''"""메인 모듈"""

from utils import Calculator

def main():
    """프로그램 시작점"""
    calc = Calculator()
    result = calc.add(10, 20)
    print(f"결과: {result}")
    return result

if __name__ == "__main__":
    main()
''',
        encoding="utf-8",
    )

    utils_py = tmp_path / "utils.py"
    utils_py.write_text(
        '''"""유틸리티 모듈"""

class Calculator:
    """간단한 계산기 클래스"""
    
    def add(self, a: int, b: int) -> int:
        """두 수를 더합니다"""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """두 수를 곱합니다"""
        return a * b

def helper_function():
    """도우미 함수"""
    return "helper"
''',
        encoding="utf-8",
    )

    return tmp_path


def test_e2e_code_analysis_and_search(test_config: CoreConfig, sample_code_files: Path):
    """E2E 테스트: 코드 분석부터 GraphRAG 검색까지"""

    logger.info("E2E 테스트 시작")

    # 컴포넌트를 함수 범위에서 초기화
    persistence = None

    try:
        # 1. 모든 컴포넌트 초기화
        logger.info("컴포넌트 초기화 중...")
        persistence, embedder, retriever, graph_service, agent = create_from_config(
            test_config
        )

        # 2. 프로젝트 데이터 정리 (이전 테스트 결과 삭제)
        logger.info("이전 테스트 데이터 정리 중...")
        persistence.clear_project_data(test_config.project_name)

        # 3. 샘플 코드 분석 및 저장
        logger.info(f"코드 분석 시작: {sample_code_files}")
        graph = graph_service.analyze_and_store_project(
            str(sample_code_files), create_embeddings=True
        )

        # 4. 분석 결과 검증
        logger.info("분석 결과 검증 중...")
        assert len(graph.nodes) > 0, "노드가 생성되지 않았습니다"
        assert len(graph.relations) > 0, "관계가 생성되지 않았습니다"

        # 함수와 클래스가 제대로 분석되었는지 확인
        node_names = [node.name for node in graph.nodes.values()]
        assert "Calculator" in node_names, "Calculator 클래스가 분석되지 않았습니다"
        assert "add" in node_names, "add 메서드가 분석되지 않았습니다"
        assert "main" in node_names, "main 함수가 분석되지 않았습니다"

        logger.info(
            f"분석 완료: {len(graph.nodes)}개 노드, {len(graph.relations)}개 관계"
        )

        # 5. 임베딩 검색 테스트 (Agent 없이)
        logger.info("임베딩 검색 테스트 중...")
        search_results = agent.get_search_results("계산기 클래스")

        assert len(search_results) > 0, "검색 결과가 없습니다"

        # Calculator 클래스가 상위 결과에 포함되어 있는지 확인
        top_result = search_results[0]
        assert (
            "Calculator" in top_result.name
            or "calculator" in top_result.source_code.lower()
        ), "Calculator 관련 결과가 상위에 없습니다"

        logger.info(f"검색 완료: {len(search_results)}개 결과")

        # 6. GraphRAG 질의 테스트 (1회만 - 무료 플랜 보호)
        logger.info("GraphRAG 질의 테스트 중... (Gemini 호출)")
        answer = agent.query("Calculator 클래스의 add 메서드는 무엇을 하나요?")

        assert answer is not None, "답변이 생성되지 않았습니다"
        assert len(answer) > 10, "답변이 너무 짧습니다"
        assert "add" in answer.lower(), "답변에 add 메서드 관련 내용이 없습니다"

        logger.info(f"GraphRAG 답변 생성 완료 (길이: {len(answer)})")
        logger.info(f"답변 미리보기: {answer[:100]}...")

        # 7. 통계 정보 확인
        stats = graph_service.get_statistics()
        logger.info(f"프로젝트 통계: {stats}")

        assert stats.get("total_nodes", 0) > 0, "통계에 노드 정보가 없습니다"

        logger.info("✅ E2E 테스트 성공!")

    except Exception as e:
        logger.error(f"❌ E2E 테스트 실패: {e!r}")
        raise

    finally:
        # 8. 정리
        if persistence is not None:
            logger.info("테스트 데이터 정리 중...")
            try:
                persistence.clear_project_data(test_config.project_name)
                persistence.close()
            except Exception as e:
                logger.warning(f"정리 중 오류: {e}")


if __name__ == "__main__":
    """직접 실행용"""

    # 임시 디렉토리에 샘플 코드 생성
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 샘플 파일 생성
        main_py = tmp_path / "main.py"
        main_py.write_text(
            '''"""메인 모듈"""

from utils import Calculator

def main():
    """프로그램 시작점"""
    calc = Calculator()
    result = calc.add(10, 20)
    print(f"결과: {result}")
    return result

if __name__ == "__main__":
    main()
''',
            encoding="utf-8",
        )

        utils_py = tmp_path / "utils.py"
        utils_py.write_text(
            '''"""유틸리티 모듈"""

class Calculator:
    """간단한 계산기 클래스"""
    
    def add(self, a: int, b: int) -> int:
        """두 수를 더합니다"""
        return a + b
    
    def multiply(self, a: int, b: int) -> int:
        """두 수를 곱합니다"""
        return a * b

def helper_function():
    """도우미 함수"""
    return "helper"
''',
            encoding="utf-8",
        )

        # 설정 생성
        config = CoreConfig()
        config.embedding.provider = "huggingface"
        config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.project_name = "test-project"

        # 환경변수에서 API 키 확인
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY 환경변수를 설정해주세요")
            exit(1)

        config.llm.api_key = api_key
        config.llm.model_name = "gemini-2.0-flash"

        # 테스트 실행
        try:
            test_e2e_code_analysis_and_search(config, tmp_path)
            print("✅ 테스트 성공!")
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            raise

"""단일 질의 테스트 - 전체 답변 확인"""

import logging
import os
import tempfile
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv

from src.core import CoreConfig, create_from_config

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_project(base_path: Path) -> None:
    """간단한 프로젝트 생성"""

    # 간단한 계산기 예제
    calculator_py = base_path / "calculator.py"
    calculator_py.write_text(
        dedent("""
        '''계산기 모듈'''
        from typing import Union
        import logging

        logger = logging.getLogger(__name__)


        class Calculator:
            '''고급 계산기 클래스'''
            
            def __init__(self):
                self.history = []
                logger.info("계산기 초기화됨")
            
            def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
                '''두 수를 더합니다
                
                Args:
                    a: 첫 번째 수
                    b: 두 번째 수
                    
                Returns:
                    두 수의 합
                    
                Raises:
                    TypeError: 입력이 숫자가 아닌 경우
                '''
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    raise TypeError("숫자만 입력 가능합니다")
                
                result = a + b
                self.history.append(f"{a} + {b} = {result}")
                logger.info(f"덧셈 수행: {a} + {b} = {result}")
                return result
            
            def multiply(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
                '''두 수를 곱합니다'''
                if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                    raise TypeError("숫자만 입력 가능합니다")
                
                result = a * b
                self.history.append(f"{a} × {b} = {result}")
                logger.info(f"곱셈 수행: {a} × {b} = {result}")
                return result
            
            def get_history(self) -> list:
                '''계산 기록 반환'''
                return self.history.copy()
            
            def clear_history(self) -> None:
                '''계산 기록 초기화'''
                self.history.clear()
                logger.info("계산 기록 초기화됨")


        def create_calculator() -> Calculator:
            '''계산기 인스턴스 생성 팩토리 함수'''
            return Calculator()


        if __name__ == "__main__":
            calc = create_calculator()
            result1 = calc.add(10, 20)
            result2 = calc.multiply(5, 3)
            print(f"덧셈 결과: {result1}")
            print(f"곱셈 결과: {result2}")
            print(f"계산 기록: {calc.get_history()}")
    """),
        encoding="utf-8",
    )


def test_single_query():
    """단일 질의 테스트"""

    logger.info("단일 질의 테스트 시작")

    # 설정 생성
    config = CoreConfig()
    config.embedding.provider = "huggingface"
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.project_name = "single-query-test"

    # 환경변수에서 API 키 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("❌ GOOGLE_API_KEY 환경변수를 설정해주세요")
        return

    config.llm.api_key = api_key
    config.llm.model_name = "gemini-2.0-flash-lite"
    config.llm.temperature = 0.1
    config.llm.max_tokens = 4096  # 긴 답변 허용

    # 컴포넌트 초기화
    persistence, embedder, retriever, graph_service, agent = create_from_config(config)

    try:
        # 임시 디렉토리에 간단한 프로젝트 생성
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            create_simple_project(tmp_path)

            # 이전 테스트 데이터 정리
            logger.info("이전 테스트 데이터 정리 중...")
            persistence.clear_project_data(config.project_name)

            # 코드 분석 및 저장
            logger.info(f"코드 분석 시작: {tmp_path}")
            graph = graph_service.analyze_and_store_project(
                str(tmp_path), create_embeddings=True
            )

            # 분석 결과 통계
            logger.info(
                f"분석 완료: {len(graph.nodes)}개 노드, {len(graph.relations)}개 관계"
            )

            # GraphRAG 질의 테스트
            query = "Calculator 클래스의 add 메서드는 어떤 기능을 하고, 어떤 매개변수를 받으며, 어떤 예외를 발생시키나요? 구체적으로 설명해주세요."

            logger.info(f"\n=== 질의: {query} ===")

            # 검색 단계 먼저 확인
            search_results = agent.get_search_results(query)
            logger.info(f"검색 결과: {len(search_results)}개")

            for idx, result in enumerate(search_results):
                result_type = getattr(result.node_type, "value", str(result.node_type))
                logger.info(f"  {idx + 1}. {result_type}: {result.name}")

            # GraphRAG 답변 생성
            logger.info("\n검색된 컨텍스트로 답변 생성 중...")
            answer = agent.query(query)

            # 답변 길이가 긴 이유 분석
            logger.info("\n=== 답변 분석 ===")
            logger.info(f"답변 총 길이: {len(answer)}자")
            logger.info(f"답변 단어 수: {len(answer.split())}개")
            logger.info(f"답변 줄 수: {len(answer.split(chr(10)))}줄")

            # 전체 답변 출력
            logger.info("\n" + "=" * 80)
            logger.info("🤖 GEMINI 2.0 FLASH LITE 전체 답변:")
            logger.info("=" * 80)
            logger.info(answer)
            logger.info("=" * 80)

            # 설정 정보 출력
            logger.info("\n=== 설정 정보 ===")
            logger.info(f"모델: {config.llm.model_name}")
            logger.info(f"최대 토큰: {config.llm.max_tokens}")
            logger.info(f"온도: {config.llm.temperature}")

            logger.info("✅ 단일 질의 테스트 완료!")

    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        raise

    finally:
        # 정리
        logger.info("테스트 데이터 정리 중...")
        try:
            persistence.clear_project_data(config.project_name)
            persistence.close()
        except Exception as e:
            logger.warning(f"정리 중 오류: {e}")


if __name__ == "__main__":
    test_single_query()

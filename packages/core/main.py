"""
코드 분석 메인 실행 파일
전체 파이프라인을 실행하여 코드를 분석하고 그래프를 생성합니다.
"""

import os
import sys
import logging
from pathlib import Path

from packages.parser.src.code_analyzer import CodeAnalyzer
from graph.src.adapter import ParserToGraphAdapter
from graph.src.models import CodeGraph
from graph.src.persistence import Neo4jPersistence
from src.embedding_service import EmbeddingService
from src.code_vectorizer import CodeVectorizer
from src.graph_rag import GraphRAGService, RAGConfig

# 패키지 경로 추가 후 import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages"))


class CodeAnalysisOrchestrator:
    """코드 분석 전체 파이프라인 오케스트레이터"""

    def __init__(self, project_path: str, project_name: str | None = None):
        self.project_path = Path(project_path)
        self.project_name = project_name or self.project_path.name

        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # 서비스 초기화
        self.neo4j_persistence: Neo4jPersistence | None = None
        self.embedding_service = None
        self.code_vectorizer = None
        self.graph_rag_service = None

        self._init_services()

    def _init_services(self):
        """서비스 초기화"""
        try:
            # Neo4j 지속성 계층
            self.neo4j_persistence = Neo4jPersistence()

            # 임베딩 서비스
            self.embedding_service = EmbeddingService()

            # 코드 벡터화 서비스
            self.code_vectorizer = CodeVectorizer(
                neo4j_persistence=self.neo4j_persistence,
                embedding_service=self.embedding_service,
            )

            # GraphRAG 서비스
            self.graph_rag_service = GraphRAGService(
                neo4j_persistence=self.neo4j_persistence,
                embedding_service=self.embedding_service,
                config=RAGConfig(
                    max_results=10, similarity_threshold=0.7, context_depth=2
                ),
            )

            self.logger.info("모든 서비스 초기화 완료")

        except Exception as e:
            self.logger.error(f"서비스 초기화 실패: {e}")
            raise

    def run_full_analysis(self, force_update: bool = False) -> bool:
        """전체 분석 파이프라인 실행"""
        try:
            self.logger.info(f"코드 분석 시작: {self.project_name}")

            # 1. Neo4j 연결 및 초기화
            if not self._setup_neo4j():
                return False

            # 2. 코드 파싱 및 분석
            code_graph = self._parse_code()
            if not code_graph:
                return False

            # 3. 기존 데이터 삭제 (force_update인 경우)
            if force_update:
                self.neo4j_persistence.clear_project_data(self.project_name)

            # 4. Neo4j에 그래프 데이터 저장
            if not self._save_to_neo4j(code_graph):
                return False

            # 5. 코드 블록 벡터화
            if not self._vectorize_code(force_update):
                return False

            # 6. 결과 출력
            self._print_results()

            self.logger.info("전체 분석 파이프라인 완료")
            return True

        except Exception as e:
            self.logger.error(f"분석 파이프라인 실패: {e}")
            return False
        finally:
            if self.neo4j_persistence:
                self.neo4j_persistence.close()

    def _setup_neo4j(self) -> bool:
        """Neo4j 설정 및 연결"""
        try:
            # 연결
            if not self.neo4j_persistence.connect():
                self.logger.error("Neo4j 연결 실패")
                return False

            # 제약 조건 및 인덱스 생성
            self.neo4j_persistence.create_constraints_and_indexes()

            return True

        except Exception as e:
            self.logger.error(f"Neo4j 설정 실패: {e}")
            return False

    def _parse_code(self) -> CodeGraph:
        """코드 파싱 및 그래프 생성"""
        try:
            # 코드 분석기 초기화
            analyzer = CodeAnalyzer()

            # 코드 분석 실행
            analysis_results = analyzer.analyze_directory(str(self.project_path))

            if not analysis_results:
                self.logger.error("코드 분석 결과가 없습니다")
                return None

            # 파서 결과를 그래프 모델로 변환
            adapter = ParserToGraphAdapter()
            code_graph = adapter.convert_to_graph(
                analysis_results,
                project_name=self.project_name,
                project_path=str(self.project_path),
            )

            self.logger.info(
                f"코드 그래프 생성 완료: 노드 {len(code_graph.nodes)}개, 관계 {len(code_graph.relations)}개"
            )
            return code_graph

        except Exception as e:
            self.logger.error(f"코드 파싱 실패: {e}")
            return None

    def _save_to_neo4j(self, code_graph: CodeGraph) -> bool:
        """Neo4j에 그래프 데이터 저장"""
        try:
            # Neo4j Persistence를 사용하여 전체 그래프 저장
            if not self.neo4j_persistence.save_code_graph(
                code_graph, project_name=self.project_name
            ):
                return False

            self.logger.info("Neo4j 그래프 저장 완료")
            return True

        except Exception as e:
            self.logger.error(f"Neo4j 저장 실패: {e}")
            return False

    def _vectorize_code(self, force_update: bool) -> bool:
        """코드 블록 벡터화"""
        try:
            success = self.code_vectorizer.vectorize_project_nodes(
                project_name=self.project_name, force_update=force_update
            )

            if success:
                # 벡터화 통계 출력
                stats = self.code_vectorizer.get_vectorization_statistics(
                    self.project_name
                )
                self.logger.info(f"벡터화 완료: {stats}")

            return success

        except Exception as e:
            self.logger.error(f"벡터화 실패: {e}")
            return False

    def _print_results(self):
        """결과 출력"""
        try:
            # 데이터베이스 통계
            stats = self.neo4j_persistence.get_database_statistics()

            print("\n" + "=" * 50)
            print("📊 코드 분석 및 GraphRAG 처리 완료")
            print("=" * 50)

            print(f"🎯 프로젝트: {self.project_name}")
            print(f"📂 경로: {self.project_path}")

            print("\n📈 Neo4j 데이터베이스 통계:")
            print(f"  • 총 노드: {stats.get('total_nodes', 0)}개")
            print(f"  • 총 관계: {stats.get('total_relationships', 0)}개")

            print("\n🏷️ 노드 타입별 분포:")
            for node_type, count in stats.get("node_types", {}).items():
                print(f"  • {node_type}: {count}개")

            print("\n🔗 관계 타입별 분포:")
            for rel_type, count in stats.get("relation_types", {}).items():
                print(f"  • {rel_type}: {count}개")

            # 벡터화 통계
            vectorization_stats = self.code_vectorizer.get_vectorization_statistics(
                self.project_name
            )
            print("\n🤖 벡터화 통계:")
            print(
                f"  • 벡터화된 노드: {vectorization_stats.get('vectorized_nodes', 0)}개"
            )
            print(
                f"  • 진행률: {vectorization_stats.get('vectorization_progress', 0):.1f}%"
            )
            print(
                f"  • 임베딩 모델: {vectorization_stats.get('embedding_service', {}).get('model_name', 'N/A')}"
            )

            print("\n✅ GraphRAG 시스템 준비 완료!")
            print("   이제 코드 검색 및 컨텍스트 생성이 가능합니다.")

        except Exception as e:
            self.logger.error(f"결과 출력 실패: {e}")

    def search_code(self, query: str) -> dict:
        """코드 검색 데모"""
        try:
            if not self.graph_rag_service:
                return {"error": "GraphRAG 서비스가 초기화되지 않았습니다"}

            # Neo4j 연결 확인
            if not self.neo4j_persistence.driver:
                self.neo4j_persistence.connect()

            results = self.graph_rag_service.get_enriched_context(
                query=query, project_name=self.project_name
            )

            return results

        except Exception as e:
            self.logger.error(f"코드 검색 실패: {e}")
            return {"error": str(e)}


def main():
    """메인 실행 함수"""
    print("🚀 코드 분석 및 GraphRAG 시스템")
    print("=" * 40)

    # 환경 변수 확인
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("❌ Neo4j 환경 변수가 설정되지 않았습니다:")
        print("   NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        print("\n예시:")
        print('   export NEO4J_URI="neo4j://localhost:7687"')
        print('   export NEO4J_USER="neo4j"')
        print('   export NEO4J_PASSWORD="your_password"')
        return

    # 프로젝트 경로 설정
    default_path = Path(__file__).parent.parent.parent / "parser" / "example_code"
    project_path = input(f"분석할 프로젝트 경로 (기본값: {default_path}): ").strip()

    if not project_path:
        project_path = default_path
    else:
        project_path = Path(project_path)

    if not project_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {project_path}")
        return

    # 프로젝트 이름
    project_name = input(f"프로젝트 이름 (기본값: {project_path.name}): ").strip()
    if not project_name:
        project_name = project_path.name

    # 강제 업데이트 여부
    force_update = (
        input("기존 데이터를 삭제하고 다시 분석할까요? (y/N): ").strip().lower() == "y"
    )

    try:
        # 분석 실행
        orchestrator = CodeAnalysisOrchestrator(
            project_path=str(project_path), project_name=project_name
        )

        success = orchestrator.run_full_analysis(force_update=force_update)

        if success:
            # 검색 데모
            print("\n" + "=" * 50)
            print("🔍 코드 검색 데모")
            print("=" * 50)

            while True:
                query = input("\n검색 쿼리 입력 (종료: 'quit'): ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    break

                if query:
                    print("\n🔍 검색 중...")
                    results = orchestrator.search_code(query)

                    if "error" in results:
                        print(f"❌ 오류: {results['error']}")
                    else:
                        print(f"\n📝 {results['summary']}")

                        for i, match in enumerate(results["matches"][:3], 1):
                            node = match["node"]
                            print(f"\n{i}. {node['name']} ({node['type']})")
                            print(f"   📁 {node['file_path']}")
                            print(
                                f"   📏 줄: {node.get('start_line', 'N/A')}-{node.get('end_line', 'N/A')}"
                            )
                            print(f"   🎯 유사도: {node.get('score', 0):.3f}")

        print("\n👋 프로그램을 종료합니다.")

    except KeyboardInterrupt:
        print("\n\n👋 사용자가 프로그램을 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")


if __name__ == "__main__":
    main()

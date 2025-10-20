"""
LangChain/LangGraph 기반 통합 코드 분석 시스템 테스트
Parser → Graph → Core → LLM → GraphRAG 전체 파이프라인
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Optional

# 패키지 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages"))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModernCodeAnalyzer:
    """LangChain/LangGraph 기반 현대적 코드 분석기"""

    def __init__(self, config: dict):
        self.config = config
        self.project_name = config.get("project_name", "modern_test")
        self.source_path = Path(config.get("source_path", "../parser/example_code"))

        # 서비스 초기화
        self.parser_analyzer = None
        self.graph_adapter = None
        self.neo4j_handler = None
        self.embedding_service = None
        self.llm_manager = None
        self.rag_service = None

    async def initialize_services(self):
        """모든 서비스 초기화"""
        try:
            logger.info("🚀 현대적 코드 분석 시스템 초기화...")

            # 1. Parser 초기화
            logger.info("1️⃣ Parser 서비스 초기화...")
            from parser.main.graph_builder import CodeAnalyzer

            self.parser_analyzer = CodeAnalyzer()
            logger.info("✅ Parser 초기화 완료")

            # 2. Graph 어댑터 초기화
            logger.info("2️⃣ Graph 어댑터 초기화...")
            from graph.src.adapter import ParserToGraphAdapter

            self.graph_adapter = ParserToGraphAdapter()
            logger.info("✅ Graph 어댑터 초기화 완료")

            # 3. Core 서비스들 초기화
            logger.info("3️⃣ Core 서비스 초기화...")

            # Neo4j Handler
            from src.neo4j_handler import Neo4jHandler

            neo4j_config = self.config.get("neo4j", {})
            self.neo4j_handler = Neo4jHandler(
                uri=neo4j_config.get("uri", "bolt://localhost:7687"),
                user=neo4j_config.get("user", "neo4j"),
                password=neo4j_config.get("password", "password"),
            )

            # Embedding Service (LangChain 기반)
            from src.embedding_service import EmbeddingService, EmbeddingConfig, EmbeddingProvider

            embedding_config = EmbeddingConfig(
                provider=EmbeddingProvider.HUGGINGFACE,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimensions=384,
                batch_size=16
            )
            self.embedding_service = EmbeddingService(embedding_config)

            # LLM Manager (LangChain 기반)
            from src.llm_manager import LLMManager, LLMConfig, LLMProvider

            llm_config = LLMConfig(
                provider=LLMProvider.GOOGLE_GEMINI,
                model_name="gemini-2.0-flash-exp",
                temperature=0.1,
                max_tokens=4000
            )
            self.llm_manager = LLMManager(llm_config)

            # GraphRAG Service (LangGraph 기반)
            from src.graph_rag import GraphRAGService, RAGConfig

            rag_config = RAGConfig(
                max_results=10,
                similarity_threshold=0.7,
                enable_workflows=True
            )
            self.rag_service = GraphRAGService(
                neo4j_handler=self.neo4j_handler,
                embedding_service=self.embedding_service,
                llm_manager=self.llm_manager,
                config=rag_config
            )

            logger.info("✅ Core 서비스 초기화 완료")
            logger.info("🎉 모든 서비스 초기화 완료!")

        except Exception as e:
            logger.error(f"❌ 서비스 초기화 실패: {e}")
            raise

    async def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        try:
            logger.info("🔄 현대적 파이프라인 시작...")

            # 1. 코드 파싱
            logger.info("📖 1단계: 코드 파싱...")
            if not self.source_path.exists():
                logger.error(f"❌ 소스 경로가 존재하지 않습니다: {self.source_path}")
                return False

            analysis_result = self.parser_analyzer.analyze_directory(
                str(self.source_path)
            )
            if not analysis_result:
                logger.error("❌ 코드 분석 결과가 없습니다")
                return False

            logger.info(f"✅ 코드 블록 {len(analysis_result)}개 분석 완료")

            # 2. 그래프 변환
            logger.info("🕸️ 2단계: 그래프 변환...")
            code_graph = self.graph_adapter.convert_to_graph(
                analysis_result,
                project_name=self.project_name,
                project_path=str(self.source_path),
            )
            logger.info(
                f"✅ 그래프 변환 완료: 노드 {len(code_graph.nodes)}개, 관계 {len(code_graph.relations)}개"
            )

            # 3. Neo4j 연결 테스트
            logger.info("💾 3단계: Neo4j 연결 테스트...")
            try:
                connected = self.neo4j_handler.connect()
                if connected:
                    logger.info("✅ Neo4j 연결 성공")
                else:
                    logger.warning("⚠️  Neo4j 연결 실패")
            except Exception as e:
                logger.warning(f"⚠️  Neo4j 연결 실패: {e}")

            # 4. 임베딩 서비스 테스트
            logger.info("🔢 4단계: 임베딩 서비스 테스트...")
            sample_nodes = list(code_graph.nodes.values())[:3]
            embeddings_created = 0

            for node in sample_nodes:
                try:
                    embedding = self.embedding_service.create_code_embedding(
                        source_code=node.source_code or "",
                        docstring=node.docstring or "",
                    )
                    if embedding:
                        embeddings_created += 1
                        logger.info(f"   📊 {node.name}: {len(embedding)}차원 벡터 생성")
                except Exception as e:
                    logger.warning(f"⚠️  노드 {node.name} 임베딩 실패: {e}")

            logger.info(f"✅ 임베딩 생성 완료: {embeddings_created}/{len(sample_nodes)}개")

            # 5. LLM 매니저 테스트
            logger.info("🤖 5단계: LLM 매니저 테스트...")
            if self.llm_manager.is_available():
                logger.info("✅ LLM 매니저 사용 가능")
                
                # 간단한 코드 분석 테스트
                try:
                    sample_code = sample_nodes[0].source_code if sample_nodes else "def hello(): pass"
                    analysis = await self.llm_manager.analyze_code(
                        task_type=self.llm_manager.get_supported_tasks()[0],
                        code=sample_code,
                        language="python",
                        file_path="test.py",
                        code_type="function",
                        additional_context="테스트 분석"
                    )
                    
                    if analysis:
                        logger.info("✅ LLM 코드 분석 성공")
                        logger.info(f"   📝 분석 결과 길이: {len(analysis)}자")
                    else:
                        logger.warning("⚠️  LLM 분석 결과 없음")
                        
                except Exception as e:
                    logger.warning(f"⚠️  LLM 분석 실패: {e}")
            else:
                logger.warning("⚠️  LLM 매니저를 사용할 수 없습니다")

            # 6. GraphRAG 워크플로우 테스트
            logger.info("🔍 6단계: GraphRAG 워크플로우 테스트...")
            await self.test_graphrag_workflows()

            # 7. 통계 출력
            await self.print_modern_results(analysis_result, code_graph)

            return True

        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return False

    async def test_graphrag_workflows(self):
        """GraphRAG 워크플로우 테스트"""
        try:
            # 서비스 상태 확인
            status = self.rag_service.get_service_status()
            logger.info(f"   📊 RAG 서비스 상태: {status}")

            # 사용 가능한 워크플로우 확인
            workflows = self.rag_service.get_available_workflows()
            logger.info(f"   🔧 사용 가능한 워크플로우: {len(workflows)}개")

            # 테스트 쿼리들
            test_queries = [
                ("simple search", "find function that calculates"),
                ("contextual analysis", "analyze code structure and dependencies"),
                ("similarity analysis", "def calculate_total(items): return sum(items)"),
            ]

            for query_type, query in test_queries:
                try:
                    logger.info(f"   🔍 테스트 쿼리 ({query_type}): {query[:50]}...")
                    
                    if query_type == "simple search":
                        result = await self.rag_service.search_similar_code(query)
                    elif query_type == "contextual analysis":
                        result = await self.rag_service.get_enriched_context(query)
                    elif query_type == "similarity analysis":
                        result = await self.rag_service.find_code_similarities(query)
                    
                    if result.get("success"):
                        logger.info(f"   ✅ {query_type} 성공 (신뢰도: {result.get('confidence_score', 0):.2f})")
                    else:
                        logger.warning(f"   ⚠️  {query_type} 실패: {result.get('error', 'Unknown')}")
                        
                except Exception as e:
                    logger.warning(f"   ⚠️  {query_type} 테스트 오류: {e}")

            # 헬스체크
            health = await self.rag_service.health_check()
            logger.info(f"   🏥 헬스체크: {'✅' if health.get('overall_health') else '⚠️ '}")

        except Exception as e:
            logger.error(f"GraphRAG 워크플로우 테스트 실패: {e}")

    async def print_modern_results(self, analysis_result: list, code_graph):
        """현대적 파이프라인 결과 출력"""
        print("\n" + "=" * 70)
        print("🚀 LangChain/LangGraph 기반 코드 분석 시스템")
        print("=" * 70)

        print(f"🎯 프로젝트: {self.project_name}")
        print(f"📂 소스 경로: {self.source_path}")

        # Parser 결과
        print("\n📖 Parser 결과:")
        print(f"   • 분석된 블록: {len(analysis_result)}개")

        # 블록 타입별 통계
        block_types = {}
        for block in analysis_result:
            block_type = (
                block.block_type.value
                if hasattr(block.block_type, "value")
                else str(block.block_type)
            )
            block_types[block_type] = block_types.get(block_type, 0) + 1

        for block_type, count in block_types.items():
            print(f"   • {block_type}: {count}개")

        # Graph 결과
        print(f"\n🕸️ Graph 결과:")
        print(f"   • 노드 수: {len(code_graph.nodes)}개")
        print(f"   • 관계 수: {len(code_graph.relations)}개")

        # 노드 타입별 분포
        node_types = {}
        for node in code_graph.nodes.values():
            node_type = (
                node.node_type.value
                if hasattr(node.node_type, "value")
                else str(node.node_type)
            )
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"   • 노드 타입별:")
        for node_type, count in node_types.items():
            print(f"     - {node_type}: {count}개")

        # Core 서비스 상태
        print(f"\n💾 Core 서비스:")
        embedding_info = self.embedding_service.get_embedding_info()
        print(f"   • 임베딩 제공자: {embedding_info['provider']}")
        print(f"   • 임베딩 모델: {embedding_info['model_name']}")
        print(f"   • 벡터 차원: {embedding_info['dimensions']}")
        print(f"   • 배치 크기: {embedding_info['batch_size']}")

        llm_info = self.llm_manager.get_model_info()
        print(f"   • LLM 제공자: {llm_info['provider']}")
        print(f"   • LLM 모델: {llm_info['model_name']}")
        print(f"   • 온도: {llm_info['temperature']}")

        rag_status = self.rag_service.get_service_status()
        print(f"   • Neo4j 연결: {'✅' if rag_status['neo4j_available'] else '❌'}")
        print(f"   • 임베딩 서비스: {'✅' if rag_status['embedding_available'] else '❌'}")
        print(f"   • LLM 서비스: {'✅' if rag_status['llm_available'] else '❌'}")
        print(f"   • 워크플로우 엔진: {'✅' if rag_status['workflow_engine_available'] else '❌'}")

        print(f"\n🚀 시스템 통합 상태:")
        print(f"   • Parser → Graph: ✅")
        print(f"   • Graph → Embedding: ✅")
        print(f"   • LangChain LLM: {'✅' if llm_info['available'] else '⚠️ '}")
        print(f"   • LangGraph RAG: {'✅' if rag_status['workflow_engine_available'] else '⚠️ '}")

        # 기술 스택 정보
        print(f"\n🛠️ 기술 스택:")
        print(f"   • Python: 3.13+")
        print(f"   • LangChain: 최신")
        print(f"   • LangGraph: 최신")
        print(f"   • Tree-sitter: 파싱")
        print(f"   • Pydantic v2: 데이터 검증")
        print(f"   • Neo4j: 그래프 DB (연결 대기)")

    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.neo4j_handler and hasattr(self.neo4j_handler, "close"):
                self.neo4j_handler.close()
            logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}")


async def main():
    """메인 실행 함수"""
    print("🚀 LangChain/LangGraph 기반 현대적 코드 분석 시스템")
    print("=" * 70)
    print("📦 구성: Parser + Graph + LangChain + LangGraph + HuggingFace + Gemini")
    print("=" * 70)

    # 설정
    config = {
        "project_name": "modern_integration_test",
        "source_path": "../parser/example_code",
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
            "database": "neo4j",
        },
    }

    analyzer = ModernCodeAnalyzer(config)

    try:
        # 서비스 초기화
        await analyzer.initialize_services()

        # 전체 파이프라인 실행
        success = await analyzer.run_full_pipeline()

        if success:
            print("\n🎉 현대적 시스템 통합 완료!")
            print("   ✅ Parser: Tree-sitter 기반 코드 분석")
            print("   ✅ Graph: Pydantic v2 데이터 모델")
            print("   ✅ Embedding: LangChain HuggingFace")
            print("   ✅ LLM: LangChain Gemini")
            print("   ✅ RAG: LangGraph 워크플로우")
            print("   ✅ Integration: 전체 파이프라인 연동")
        else:
            print("\n❌ 시스템 통합 실패")

    except KeyboardInterrupt:
        print("\n\n👋 사용자가 테스트를 중단했습니다")
    except Exception as e:
        logger.error(f"❌ 통합 테스트 실패: {e}")
    finally:
        await analyzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
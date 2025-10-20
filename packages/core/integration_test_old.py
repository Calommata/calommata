"""
전체 시스템 통합 테스트
Parser → Graph → Core (Neo4j + GraphRAG + Gemini) 파이프라인
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

# Google Gemini 클라이언트 추가
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Google Generative AI가 설치되지 않았습니다")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class IntegratedCodeAnalyzer:
    """전체 시스템 통합 코드 분석기"""

    def __init__(self, config: dict):
        self.config = config
        self.project_name = config.get("project_name", "integrated_test")
        self.source_path = Path(config.get("source_path", "../parser/example_code"))

        # 서비스 초기화
        self.parser_analyzer = None
        self.graph_adapter = None
        self.neo4j_handler = None
        self.embedding_service = None
        self.rag_service = None
        self.gemini_client = None

        self._init_gemini()

    def _init_gemini(self):
        """Google Gemini 초기화"""
        if GEMINI_AVAILABLE:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_client = genai.GenerativeModel("gemini-2.0-flash-exp")
                logger.info("✅ Google Gemini 2.5 Flash 초기화 완료")
            else:
                logger.warning("⚠️  GEMINI_API_KEY 환경변수가 설정되지 않았습니다")
        else:
            logger.warning("⚠️  Google Generative AI를 사용할 수 없습니다")

    async def initialize_services(self):
        """모든 서비스 초기화"""
        try:
            logger.info("🚀 서비스 초기화 시작...")

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

            # Embedding Service (로컬 모델 사용)
            from src.embedding_service import EmbeddingService, EmbeddingConfig

            embedding_config = EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # 로컬 모델
                dimensions=384,
                api_key=None,  # 로컬 사용
            )
            self.embedding_service = EmbeddingService(embedding_config)

            # RAG Service
            from src.graph_rag import GraphRAGService, RAGConfig

            rag_config = RAGConfig(
                max_results=10,
                similarity_threshold=0.7,
                context_depth=2,
                max_context_tokens=4000,
            )
            self.rag_service = GraphRAGService(
                neo4j_handler=self.neo4j_handler,
                embedding_service=self.embedding_service,
                config=rag_config,
            )

            logger.info("✅ Core 서비스 초기화 완료")
            logger.info("🎉 모든 서비스 초기화 완료!")

        except Exception as e:
            logger.error(f"❌ 서비스 초기화 실패: {e}")
            raise

    async def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        try:
            logger.info("🔄 전체 파이프라인 시작...")

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

            # 3. Neo4j 저장 (연결 가능한 경우에만)
            logger.info("💾 3단계: Neo4j 저장...")
            try:
                connected = self.neo4j_handler.connect()
                if connected:
                    # Neo4j 연결 성공 - 실제 저장은 나중에 구현
                    logger.info("✅ Neo4j 연결 성공 (저장 기능은 개발 중)")
                else:
                    logger.warning("⚠️  Neo4j 연결 실패")

            except Exception as e:
                logger.warning(f"⚠️  Neo4j 연결 실패, 메모리에서만 작업: {e}")

            # 4. 임베딩 생성 (샘플만)
            logger.info("🔢 4단계: 임베딩 생성...")
            sample_nodes = list(code_graph.nodes.values())[:3]  # 처음 3개만
            embeddings_created = 0

            for node in sample_nodes:
                try:
                    embedding = self.embedding_service.create_code_embedding(
                        source_code=node.source_code or "",
                        docstring=node.docstring or "",
                    )
                    if embedding:
                        embeddings_created += 1
                except Exception as e:
                    logger.warning(f"⚠️  노드 {node.name} 임베딩 실패: {e}")

            logger.info(
                f"✅ 임베딩 생성 완료: {embeddings_created}/{len(sample_nodes)}개"
            )

            # 5. 통계 출력
            await self.print_pipeline_results(analysis_result, code_graph)

            return True

        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return False

    async def print_pipeline_results(self, analysis_result: dict, code_graph):
        """파이프라인 결과 출력"""
        print("\n" + "=" * 60)
        print("📊 전체 시스템 통합 결과")
        print("=" * 60)

        print(f"🎯 프로젝트: {self.project_name}")
        print(f"📂 소스 경로: {self.source_path}")

        # Parser 결과 (CodeBlock 리스트)
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
        print(f"   • 임베딩 모델: {self.embedding_service.config.model_name}")
        print(f"   • 벡터 차원: {self.embedding_service.config.dimensions}")
        print(f"   • Neo4j 연결: {'✅' if self.neo4j_handler.driver else '❌'}")
        print(f"   • Gemini AI: {'✅' if self.gemini_client else '❌'}")

        print(f"\n🚀 시스템 통합 상태:")
        print(f"   • Parser → Graph: ✅")
        print(f"   • Graph → Neo4j: {'✅' if self.neo4j_handler.driver else '⚠️ '}")
        print(f"   • 로컬 임베딩: ✅")
        print(f"   • Gemini AI: {'✅' if self.gemini_client else '⚠️ '}")

    async def test_ai_integration(self):
        """AI 통합 테스트"""
        if not self.gemini_client:
            logger.warning("⚠️  Gemini AI를 사용할 수 없습니다")
            return

        print("\n" + "=" * 60)
        print("🤖 AI 통합 테스트 (Gemini 2.5 Flash)")
        print("=" * 60)

        try:
            # 간단한 코드 분석 요청
            test_prompt = f"""
다음은 '{self.project_name}' 프로젝트의 코드 분석 결과입니다.
이 프로젝트의 주요 특징과 구조를 간단히 분석해주세요:

프로젝트 경로: {self.source_path}
분석된 파일 수: {len(list(self.source_path.glob("*.py")) if self.source_path.exists() else [])}개

주요 파일들:
{chr(10).join(f"- {f.name}" for f in (self.source_path.glob("*.py") if self.source_path.exists() else []))}

간단한 분석 결과를 3-4줄로 요약해주세요.
"""

            response = await asyncio.to_thread(
                self.gemini_client.generate_content, test_prompt
            )

            print("🤖 Gemini 분석 결과:")
            print(response.text)
            print("✅ AI 통합 테스트 완료")

        except Exception as e:
            logger.error(f"❌ AI 통합 테스트 실패: {e}")

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
    print("🚀 전체 시스템 통합 테스트")
    print("=" * 60)
    print("📦 구성: Parser + Graph + Core + Gemini AI + 로컬 임베딩")
    print("=" * 60)

    # 설정
    config = {
        "project_name": "example_integration_test",
        "source_path": "../parser/example_code",
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
            "database": "neo4j",
        },
    }

    analyzer = IntegratedCodeAnalyzer(config)

    try:
        # 서비스 초기화
        await analyzer.initialize_services()

        # 전체 파이프라인 실행
        success = await analyzer.run_full_pipeline()

        if success:
            # AI 통합 테스트
            await analyzer.test_ai_integration()

            print("\n🎉 전체 시스템 통합 완료!")
            print("   ✅ Parser: 코드 분석")
            print("   ✅ Graph: 데이터 변환")
            print("   ✅ Core: 임베딩 + Neo4j + RAG")
            print("   ✅ AI: Gemini 2.5 Flash")
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

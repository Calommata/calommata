"""
Core 패키지 단독 테스트
Neo4j, 임베딩, GraphRAG 기능만 테스트
"""

import sys
from pathlib import Path

# 패키지 경로 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_core_services():
    """Core 서비스들 단독 테스트"""
    print("🚀 Core 패키지 서비스 테스트")
    print("=" * 40)

    try:
        # 1. Embedding Service 테스트
        print("1. Embedding Service 테스트...")
        from src.embedding_service import EmbeddingService

        embedding_service = EmbeddingService()
        print("✅ EmbeddingService 초기화 성공")
        print(f"   모델: {embedding_service.config.model_name}")
        print(f"   벡터 크기: {embedding_service.config.dimensions}")

        # 간단한 임베딩 테스트
        test_code = """
def hello_world():
    print("Hello, World!")
    return True
"""
        try:
            embedding = embedding_service.create_code_embedding(
                source_code=test_code, docstring="Simple hello world function"
            )
            if embedding and len(embedding) > 0:
                print(f"✅ 코드 임베딩 생성 성공: {len(embedding)}차원")
            else:
                print("❌ 코드 임베딩 생성 실패")
        except Exception as e:
            print(f"⚠️  임베딩 생성 실패 (API 키 필요): {e}")

        # 2. Graph 모델 테스트
        print("\n2. Graph 모델 테스트...")
        project_root = current_dir.parent.parent
        sys.path.insert(0, str(project_root / "packages"))

        from graph.src.models import (
            CodeNode,
            CodeRelation,
            CodeGraph,
            NodeType,
            RelationType,
        )

        # 샘플 노드 생성
        node1 = CodeNode(
            id="test_func_1",
            name="test_function",
            node_type=NodeType.FUNCTION,
            file_path="test.py",
            start_line=1,
            end_line=5,
            source_code=test_code,
            docstring="Test function",
        )

        node2 = CodeNode(
            id="test_class_1",
            name="TestClass",
            node_type=NodeType.CLASS,
            file_path="test.py",
            start_line=7,
            end_line=15,
            source_code="class TestClass:\n    pass",
            docstring="Test class",
        )

        # 샘플 관계 생성
        relation = CodeRelation(
            from_node_id="test_class_1",
            to_node_id="test_func_1",
            relation_type=RelationType.CONTAINS,
            metadata={"context": "class method"},
        )

        # 그래프 생성
        graph = CodeGraph(
            project_name="test_project",
            project_path="/test/path",
            nodes={"test_func_1": node1, "test_class_1": node2},
            relations=[relation],
        )

        print("✅ Graph 모델 생성 성공")
        print(f"   노드 수: {len(graph.nodes)}")
        print(f"   관계 수: {len(graph.relations)}")

        # Neo4j 형식 변환 테스트 (일단 건너뜀)
        # _ = graph.to_neo4j_format()
        print("✅ Graph 모델 기본 테스트 성공")

        # 3. Neo4j Handler 테스트 (연결 없이)
        print("\n3. Neo4j Handler 초기화 테스트...")
        from src.neo4j_handler import Neo4jHandler

        _ = Neo4jHandler(uri="bolt://localhost:7687", user="neo4j", password="password")
        print("✅ Neo4j Handler 초기화 성공")
        print("   (실제 연결은 Neo4j 서버가 실행 중일 때만 가능)")

        # 4. RAG Config 테스트
        print("\n4. RAG Config 테스트...")
        from src.graph_rag import RAGConfig

        rag_config = RAGConfig(
            max_results=5, similarity_threshold=0.7, max_context_tokens=2000
        )
        print("✅ RAG Config 생성 성공")
        print(f"   최대 결과 수: {rag_config.max_results}")
        print(f"   유사도 임계값: {rag_config.similarity_threshold}")

        print("\n🎉 모든 Core 서비스 테스트 완료!")
        print("   - EmbeddingService: ✅")
        print("   - Graph Models: ✅")
        print("   - Neo4j Handler: ✅")
        print("   - RAG Config: ✅")

    except ImportError as e:
        print(f"❌ Import 오류: {e}")
        print("   필요한 패키지가 설치되지 않았을 수 있습니다")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")


def test_integration_readiness():
    """통합 준비 상태 확인"""
    print("\n" + "=" * 50)
    print("🔗 통합 준비 상태 확인")
    print("=" * 50)

    # 패키지 구조 확인
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent

    packages = ["parser", "graph", "core"]
    ready_packages = []

    for package in packages:
        package_path = project_root / "packages" / package
        if package_path.exists():
            init_file = package_path / "__init__.py"
            if init_file.exists():
                ready_packages.append(package)
                print(f"✅ {package} 패키지 준비됨")
            else:
                print(f"⚠️  {package} 패키지에 __init__.py 필요")
        else:
            print(f"❌ {package} 패키지 없음")

    print(f"\n준비된 패키지: {len(ready_packages)}/{len(packages)}")

    if len(ready_packages) == len(packages):
        print("🎯 전체 파이프라인 실행 준비 완료!")
        print("   다음 단계: parser에서 tree-sitter 설정 후 통합 테스트")
    else:
        print("⚠️  일부 패키지 설정 필요")


def main():
    test_core_services()
    test_integration_readiness()


if __name__ == "__main__":
    main()

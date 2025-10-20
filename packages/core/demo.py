"""
GraphRAG 데모 및 테스트 스크립트
Neo4j에 저장된 코드 그래프를 활용한 검색 및 추천 기능 시연
"""

import os
import sys
from pathlib import Path

# 패키지 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

from src.neo4j_handler import Neo4jHandler
from src.embedding_service import EmbeddingService
from src.graph_rag import GraphRAGService, RAGConfig


def test_connection():
    """Neo4j 연결 테스트"""
    print("🔗 Neo4j 연결 테스트...")

    with Neo4jHandler() as neo4j:
        if neo4j.connect():
            print("✅ Neo4j 연결 성공")

            # 데이터베이스 통계 조회
            stats = neo4j.get_database_statistics()
            print(f"📊 총 노드: {stats.get('total_nodes', 0)}개")
            print(f"📊 총 관계: {stats.get('total_relationships', 0)}개")

            return True
        else:
            print("❌ Neo4j 연결 실패")
            return False


def demo_code_search():
    """코드 검색 데모"""
    print("\n🔍 코드 검색 데모")
    print("=" * 40)

    # 서비스 초기화
    neo4j_handler = Neo4jHandler()
    embedding_service = EmbeddingService()

    if not neo4j_handler.connect():
        print("❌ Neo4j 연결 실패")
        return

    graph_rag = GraphRAGService(
        neo4j_handler=neo4j_handler,
        embedding_service=embedding_service,
        config=RAGConfig(max_results=5, similarity_threshold=0.5, context_depth=2),
    )

    # 검색 쿼리 예시들
    sample_queries = [
        "database connection and query execution",
        "user authentication and login",
        "API request handling",
        "class inheritance and methods",
        "error handling and exceptions",
    ]

    print("📝 샘플 검색 쿼리:")
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")

    while True:
        print("\n" + "-" * 40)
        choice = input("검색할 쿼리 번호 (1-5) 또는 직접 입력, 종료(q): ").strip()

        if choice.lower() == "q":
            break

        query = ""
        if choice.isdigit() and 1 <= int(choice) <= 5:
            query = sample_queries[int(choice) - 1]
        else:
            query = choice

        if query:
            print(f"\n🔍 검색 중: '{query}'")

            try:
                results = graph_rag.get_enriched_context(query)

                print(f"📝 {results['summary']}")

                if results["matches"]:
                    print(f"\n🎯 검색 결과 ({len(results['matches'])}개):")

                    for i, match in enumerate(results["matches"][:3], 1):
                        node = match["node"]
                        print(f"\n{i}. {node['name']} ({node['type']})")
                        print(f"   📁 {node['file_path']}")
                        print(
                            f"   📏 줄: {node.get('start_line', 'N/A')}-{node.get('end_line', 'N/A')}"
                        )
                        print(f"   🎯 유사도: {node.get('score', 0):.3f}")

                        # docstring이 있으면 출력
                        if node.get("docstring"):
                            print(f"   📖 {node['docstring'][:100]}...")

                        # 관련 함수들
                        related_funcs = match.get("related_functions", [])
                        if related_funcs:
                            print(
                                f"   🔗 관련 함수: {', '.join([f['name'] for f in related_funcs[:3]])}"
                            )
                else:
                    print("❌ 검색 결과가 없습니다.")

            except Exception as e:
                print(f"❌ 검색 오류: {e}")

    neo4j_handler.close()


def demo_code_recommendations():
    """코드 추천 데모"""
    print("\n💡 코드 추천 데모")
    print("=" * 40)

    # 서비스 초기화
    neo4j_handler = Neo4jHandler()
    embedding_service = EmbeddingService()

    if not neo4j_handler.connect():
        print("❌ Neo4j 연결 실패")
        return

    graph_rag = GraphRAGService(
        neo4j_handler=neo4j_handler, embedding_service=embedding_service
    )

    # 모든 노드 조회
    try:
        with neo4j_handler.driver.session() as session:
            result = session.run("""
                MATCH (n:CodeNode)
                WHERE n.type IN ['Class', 'Function', 'Method']
                RETURN n.id AS id, n.name AS name, n.type AS type, n.file_path AS file_path
                LIMIT 10
            """)

            nodes = [record.data() for record in result]

            if not nodes:
                print("❌ 추천할 노드가 없습니다.")
                return

            print("📋 사용 가능한 코드 블록:")
            for i, node in enumerate(nodes, 1):
                print(
                    f"  {i}. {node['name']} ({node['type']}) - {Path(node['file_path']).name}"
                )

            while True:
                choice = input(
                    f"\n추천받을 코드 번호 (1-{len(nodes)}) 또는 종료(q): "
                ).strip()

                if choice.lower() == "q":
                    break

                if choice.isdigit() and 1 <= int(choice) <= len(nodes):
                    selected_node = nodes[int(choice) - 1]
                    node_id = selected_node["id"]

                    print(f"\n🎯 '{selected_node['name']}' 관련 추천:")

                    # 여러 타입의 추천 실행
                    recommendation_types = [
                        ("similar", "유사한 코드"),
                        ("related", "관련된 코드"),
                        ("contextual", "컨텍스트 기반"),
                    ]

                    for rec_type, rec_name in recommendation_types:
                        print(f"\n📌 {rec_name}:")

                        try:
                            recommendations = graph_rag.recommend_related_code(
                                node_id=node_id, recommendation_type=rec_type
                            )

                            if recommendations:
                                for i, rec in enumerate(recommendations[:3], 1):
                                    node_rec = rec["node"]
                                    print(
                                        f"   {i}. {node_rec['name']} ({node_rec['type']})"
                                    )
                                    print(f"      이유: {rec['reason']}")
                                    print(
                                        f"      파일: {Path(node_rec['file_path']).name}"
                                    )
                            else:
                                print("   추천 결과 없음")

                        except Exception as e:
                            print(f"   오류: {e}")

                else:
                    print("❌ 올바른 번호를 입력하세요.")

    except Exception as e:
        print(f"❌ 노드 조회 오류: {e}")

    neo4j_handler.close()


def demo_dependency_analysis():
    """의존성 분석 데모"""
    print("\n🔗 의존성 분석 데모")
    print("=" * 40)

    neo4j_handler = Neo4jHandler()

    if not neo4j_handler.connect():
        print("❌ Neo4j 연결 실패")
        return

    graph_rag = GraphRAGService(
        neo4j_handler=neo4j_handler, embedding_service=EmbeddingService()
    )

    # 클래스 노드들 조회
    try:
        with neo4j_handler.driver.session() as session:
            result = session.run("""
                MATCH (n:CodeNode)
                WHERE n.type = 'Class'
                RETURN n.id AS id, n.name AS name, n.file_path AS file_path
                LIMIT 5
            """)

            classes = [record.data() for record in result]

            if not classes:
                print("❌ 분석할 클래스가 없습니다.")
                return

            print("📋 사용 가능한 클래스:")
            for i, cls in enumerate(classes, 1):
                print(f"  {i}. {cls['name']} - {Path(cls['file_path']).name}")

            while True:
                choice = input(
                    f"\n분석할 클래스 번호 (1-{len(classes)}) 또는 종료(q): "
                ).strip()

                if choice.lower() == "q":
                    break

                if choice.isdigit() and 1 <= int(choice) <= len(classes):
                    selected_class = classes[int(choice) - 1]
                    class_id = selected_class["id"]

                    print(f"\n🔍 '{selected_class['name']}' 의존성 분석:")

                    try:
                        dependencies = graph_rag.find_code_dependencies(class_id)

                        # 중심 노드 정보
                        center = dependencies.get("center_node")
                        if center:
                            print(
                                f"\n🎯 중심 노드: {center['name']} ({center['type']})"
                            )
                            print(f"   📁 {center['file_path']}")

                        # 이 클래스가 의존하는 것들
                        deps = dependencies.get("dependencies", [])
                        if deps:
                            print(f"\n⬇️  의존하는 대상 ({len(deps)}개):")
                            for dep in deps[:5]:
                                if dep["node"]:
                                    node = dep["node"]
                                    print(
                                        f"   • {node['name']} ({dep['relation_type']})"
                                    )

                        # 이 클래스에 의존하는 것들
                        dependents = dependencies.get("dependents", [])
                        if dependents:
                            print(
                                f"\n⬆️  이 클래스를 사용하는 대상 ({len(dependents)}개):"
                            )
                            for dep in dependents[:5]:
                                if dep["node"]:
                                    node = dep["node"]
                                    print(
                                        f"   • {node['name']} ({dep['relation_type']})"
                                    )

                        # 같은 파일의 형제 노드들
                        siblings = dependencies.get("siblings", [])
                        if siblings:
                            print(
                                f"\n👨‍👩‍👧‍👦 같은 파일의 다른 요소들 ({len(siblings)}개):"
                            )
                            for sibling in siblings[:5]:
                                print(f"   • {sibling['name']} ({sibling['type']})")

                    except Exception as e:
                        print(f"❌ 의존성 분석 오류: {e}")

                else:
                    print("❌ 올바른 번호를 입력하세요.")

    except Exception as e:
        print(f"❌ 클래스 조회 오류: {e}")

    neo4j_handler.close()


def main():
    """메인 데모 함수"""
    print("🚀 GraphRAG 시스템 데모")
    print("=" * 50)

    # 환경 변수 확인
    required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ 필요한 환경 변수가 설정되지 않았습니다: {missing_vars}")
        print("\n예시:")
        print('export NEO4J_URI="neo4j://localhost:7687"')
        print('export NEO4J_USER="neo4j"')
        print('export NEO4J_PASSWORD="your_password"')
        return

    # 연결 테스트
    if not test_connection():
        return

    # 데모 메뉴
    while True:
        print("\n" + "=" * 50)
        print("🎯 GraphRAG 데모 메뉴")
        print("=" * 50)
        print("1. 코드 검색 (자연어 쿼리)")
        print("2. 코드 추천 (유사/관련/컨텍스트)")
        print("3. 의존성 분석 (관계 구조)")
        print("4. 종료")

        choice = input("\n선택하세요 (1-4): ").strip()

        if choice == "1":
            demo_code_search()
        elif choice == "2":
            demo_code_recommendations()
        elif choice == "3":
            demo_dependency_analysis()
        elif choice == "4":
            print("\n👋 데모를 종료합니다.")
            break
        else:
            print("❌ 올바른 번호를 선택하세요.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 사용자가 데모를 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")

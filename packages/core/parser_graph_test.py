"""
Parser + Graph 통합 테스트
Parser의 결과를 Graph로 변환하는 테스트
"""

import sys
from pathlib import Path

# 패키지 경로 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "packages"))


def test_parser_to_graph():
    """Parser → Graph 변환 테스트"""
    print("🚀 Parser → Graph 통합 테스트")
    print("=" * 50)

    try:
        # 1. Parser 테스트
        print("1️⃣ Parser 실행...")
        parser_dir = project_root / "packages" / "parser"
        sys.path.insert(0, str(parser_dir))

        from parser.src.graph_builder import CodeAnalyzer

        analyzer = CodeAnalyzer()

        example_path = parser_dir / "example_code"
        blocks = analyzer.analyze_directory(str(example_path))

        print(f"✅ Parser 완료: {len(blocks)}개 블록 분석")

        # 블록 타입별 통계
        block_types = {}
        for block in blocks:
            block_type = (
                block.block_type.value
                if hasattr(block.block_type, "value")
                else str(block.block_type)
            )
            block_types[block_type] = block_types.get(block_type, 0) + 1

        print("   블록 타입별 분포:")
        for block_type, count in block_types.items():
            print(f"     • {block_type}: {count}개")

        # 2. Graph 변환 테스트
        print("\n2️⃣ Graph 변환...")
        from graph.src.adapter import ParserToGraphAdapter

        adapter = ParserToGraphAdapter()
        graph = adapter.convert_to_graph(
            blocks, project_name="parser_test", project_path=str(example_path)
        )

        print(f"✅ Graph 변환 완료:")
        print(f"   • 노드 수: {len(graph.nodes)}개")
        print(f"   • 관계 수: {len(graph.relations)}개")
        print(f"   • 프로젝트: {graph.project_name}")
        print(f"   • 경로: {graph.project_path}")

        # 노드 타입별 분포
        node_types = {}
        for node in graph.nodes.values():
            node_type = (
                node.node_type.value
                if hasattr(node.node_type, "value")
                else str(node.node_type)
            )
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print("\n   노드 타입별 분포:")
        for node_type, count in node_types.items():
            print(f"     • {node_type}: {count}개")

        # 샘플 노드 출력
        print("\n   샘플 노드 (처음 3개):")
        sample_nodes = list(graph.nodes.values())[:3]
        for node in sample_nodes:
            node_type_str = (
                node.node_type.value
                if hasattr(node.node_type, "value")
                else str(node.node_type)
            )
            print(
                f"     • {node.name} ({node_type_str}) - {node.file_path}:{node.start_line}"
            )

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    success = test_parser_to_graph()

    if success:
        print("\n🎉 Parser → Graph 통합 성공!")
        print("   다음 단계: Core 서비스와 통합")
    else:
        print("\n❌ Parser → Graph 통합 실패")


if __name__ == "__main__":
    main()

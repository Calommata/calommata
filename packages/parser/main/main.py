#!/usr/bin/env python3
"""
Python 코드 분석기 메인 실행 파일
Tree-sitter를 사용하여 코드 블록과 의존성을 분석하고 그래프로 시각화합니다.
"""

from graph_builder import GraphBuilder


def main():
    """메인 실행 함수"""
    print("🔍 Python Code Analyzer 시작")
    print("=" * 50)

    # 그래프 빌더 초기화
    builder = GraphBuilder()

    # 예시 코드 폴더 분석
    print("📁 코드 분석 중...")
    graph = builder.analyze_directory("example_code")

    # 통계 출력
    total_nodes = len(graph.nodes)
    total_edges = sum(len(edges) for edges in graph.edges.values())
    print("\n📊 분석 결과:")
    print(f"  - 총 노드 수: {total_nodes}")
    print(f"  - 총 의존성 수: {total_edges}")

    # 출력 파일 생성
    print("\n💾 결과 파일 생성 중...")

    # JSON 출력
    builder.export_graph("output.json")
    print("  ✓ JSON 출력: output.json")

    # HTML 시각화
    builder.visualize("graph_visualization.html")
    print("  ✓ HTML 시각화: graph_visualization.html")

    print("\n🎉 분석 완료!")
    print("브라우저에서 graph_visualization.html을 열어 결과를 확인하세요.")


if __name__ == "__main__":
    main()

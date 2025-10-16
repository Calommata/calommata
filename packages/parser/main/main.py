from graph_builder import GraphBuilder

builder = GraphBuilder()

# 예시 코드 폴더 분석
graph = builder.analyze_directory("example_code")

# 그래프 내보내기
builder.export_graph("output.json")

# 그래프 시각화
builder.visualize("graph_visualization.html")
print("✓ Graph visualization generated: graph_visualization.html")

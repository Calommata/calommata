from pathlib import Path
from base_parser import BaseParser
from ast_extractor import ASTExtractor
from import_graph import ImportGraph
import tree_sitter_python as tslanguage


class GraphBuilder:
    """코드 분석 및 import 그래프 생성"""

    def __init__(self):
        self.parser = BaseParser(tslanguage.language())
        self.extractor = ASTExtractor()
        self.graph = ImportGraph()

    def analyze_file(self, file_path: str) -> ImportGraph:
        """단일 파일 분석"""
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        tree = self.parser.parse_code(source_code)
        blocks = self.extractor.extract_blocks(tree)

        # 블록을 그래프에 추가
        for block in blocks:
            self.graph.add_block(block)

            # import 블록의 경우 의존성 추가
            if block.block_type == "import" and block.imports is not None:
                for imported_module in block.imports:
                    self.graph.add_dependency(
                        block.parent.get_full_name() if block.parent else "root",
                        imported_module,
                    )

            # 부모-자식 관계(포함 관계) 추가
            if block.parent:
                self.graph.add_dependency(
                    block.parent.get_full_name(), block.get_full_name()
                )

            # 클래스 의존성 추가 (타입 힌트에서 추출)
            if block.block_type == "class" and block.dependencies:
                for dep_class in block.dependencies:
                    # 그래프에 이미 있는 클래스 노드 찾기
                    dep_full_name = f"module.{dep_class}"
                    if dep_full_name in self.graph.nodes:
                        self.graph.add_dependency(block.get_full_name(), dep_full_name)

        return self.graph

    def analyze_directory(self, dir_path: str) -> ImportGraph:
        """디렉토리 내 모든 Python 파일 분석"""
        path = Path(dir_path)
        for py_file in path.glob("**/*.py"):
            try:
                print(str(py_file))
                self.analyze_file(str(py_file))
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")

        return self.graph

    def export_graph(self, output_path: str):
        """그래프를 JSON으로 내보내기"""
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.graph.to_dict(), f, indent=2, ensure_ascii=False)

    def visualize(self, output_path: str = "graph_visualization.html"):
        """그래프를 HTML로 시각화"""
        from graph_visualizer import GraphVisualizer

        visualizer = GraphVisualizer(self.graph)
        visualizer.generate_html(output_path)
